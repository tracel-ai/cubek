//! Scatter kernel for radix sort.
//!
//! This kernel redistributes elements to their sorted positions based on
//! the computed global offsets.
//!
//! Implementation based on b0nes164's DeviceRadixSort:
//! - Each warp/plane processes a CONTIGUOUS sub-partition of the block's data
//! - Warp 0 processes elements 0..subpart_size-1
//! - Warp 1 processes elements subpart_size..2*subpart_size-1
//! - etc.
//! - This ensures the vertical prefix sum maintains stability

use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use cubecl::prelude::*;

/// Maximum planes per block (256 threads / 32 lanes = 8 planes typical)
const MAX_PLANES: u32 = 8;

/// Scatter keys using b0nes164's algorithm with contiguous sub-partitions per warp.
///
/// Key insight: Each warp processes a CONTIGUOUS range of input elements.
/// This ensures the vertical prefix sum across warps maintains stability.
#[cube(launch_unchecked)]
pub fn scatter_keys_kernel(
    keys_in: &Tensor<u32>,
    keys_out: &mut Tensor<u32>,
    block_offsets: &Tensor<u32>,
    num_items: u32,
    pass: u32,
    #[comptime] items_per_thread: u32,
    #[comptime] _num_buckets: u32,
    #[comptime] _cube_size: u32,
) {
    // Shared memory for per-plane histograms
    let plane_hists = SharedMemory::<Atomic<u32>>::new((MAX_PLANES as usize) * NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = (CUBE_DIM + PLANE_DIM - 1) / PLANE_DIM;

    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    // Each plane processes a contiguous sub-partition
    // sub_part_size = items_per_block / num_planes = PLANE_DIM * items_per_thread
    let sub_part_size = PLANE_DIM * items_per_thread;
    let sub_part_start = block_start + plane_id * sub_part_size;

    // Initialize plane histograms to zero
    let total_hist_entries = (MAX_PLANES as usize) * NUM_BUCKETS;
    let entries_per_thread = (total_hist_entries + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..entries_per_thread {
        let idx = thread_id as usize + i * CUBE_DIM as usize;
        if idx < total_hist_entries {
            plane_hists[idx].store(0u32);
        }
    }
    sync_cube();

    // ========== PHASE 1: WLMS binning ==========
    // Each thread processes items_per_thread keys from its plane's sub-partition
    // Thread layout within plane: strided access for coalescing
    // lane 0 gets items 0, 32, 64, ...
    // lane 1 gets items 1, 33, 65, ...

    // We need to store offsets for later - use registers since we'll scatter immediately
    // after the vertical prefix sum
    let mut keys = Array::<u32>::new(items_per_thread as usize);
    let mut offsets = Array::<u32>::new(items_per_thread as usize);
    let mut valid_flags = Array::<bool>::new(items_per_thread as usize);

    #[unroll]
    for i in 0..items_per_thread {
        // Strided access within sub-partition: lane_id + i * PLANE_DIM
        let local_idx_in_subpart = lane_id + i * PLANE_DIM;
        let global_idx = sub_part_start + local_idx_in_subpart;
        let valid = global_idx < num_items;

        // Load key
        let key = select(valid, keys_in[global_idx as usize], 0u32);
        keys[i as usize] = key;
        valid_flags[i as usize] = valid;

        let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

        // WLMS: compute peer mask for threads with same digit that are valid
        let peer_mask = compute_peer_mask_valid(digit, valid, RADIX_BITS as u32);
        let rank_in_peers = count_lower_peers(peer_mask, lane_id);
        let total_peers = count_total_peers(peer_mask);
        let lowest_peer = find_lowest_peer(peer_mask);
        let is_leader = lane_id == lowest_peer && valid;

        // Leader atomically adds to this plane's histogram
        let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
        let mut base_offset = 0u32;
        if is_leader {
            base_offset = plane_hists[hist_idx].fetch_add(total_peers);
        }
        // Broadcast base offset to all peers
        base_offset = plane_shuffle(base_offset, lowest_peer);

        // This thread's offset within the plane's histogram for this digit
        offsets[i as usize] = base_offset + rank_in_peers;
    }

    sync_cube();

    // ========== PHASE 2: Vertical prefix sum across plane histograms ==========
    // For each digit, compute exclusive prefix sum across planes
    // After this, plane_hists[plane * 256 + digit] = sum of counts from planes 0..plane-1
    //
    // This is correct for stability because:
    // - Plane 0 processes global indices [block_start, block_start + sub_part_size)
    // - Plane 1 processes global indices [block_start + sub_part_size, block_start + 2*sub_part_size)
    // - So all of plane 0's elements come before plane 1's elements in input order

    if thread_id < NUM_BUCKETS as u32 {
        let digit = thread_id as usize;
        let mut running_sum = 0u32;

        #[unroll]
        for p in 0..MAX_PLANES {
            if p < num_planes {
                let idx = (p as usize) * NUM_BUCKETS + digit;
                let count = plane_hists[idx].load();
                // Store exclusive prefix (sum before this plane)
                plane_hists[idx].store(running_sum);
                running_sum += count;
            }
        }
    }

    sync_cube();

    // ========== PHASE 3: Scatter to global memory ==========
    // Final position = global_offset[digit] + plane_prefix[digit] + offset_in_plane

    #[unroll]
    for i in 0..items_per_thread {
        let valid = valid_flags[i as usize];

        if valid {
            let key = keys[i as usize];
            let offset_in_plane = offsets[i as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

            // Get the cross-plane prefix for this plane
            let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
            let plane_prefix = plane_hists[hist_idx].load();

            // Get global offset for this digit from this block
            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];

            let final_pos = global_offset + plane_prefix + offset_in_plane;
            keys_out[final_pos as usize] = key;
        }
    }
}

/// Scatter key-value pairs using the same algorithm.
#[cube(launch_unchecked)]
pub fn scatter_pairs_kernel(
    keys_in: &Tensor<u32>,
    keys_out: &mut Tensor<u32>,
    values_in: &Tensor<u32>,
    values_out: &mut Tensor<u32>,
    block_offsets: &Tensor<u32>,
    num_items: u32,
    pass: u32,
    #[comptime] items_per_thread: u32,
    #[comptime] _num_buckets: u32,
    #[comptime] _cube_size: u32,
) {
    let plane_hists = SharedMemory::<Atomic<u32>>::new((MAX_PLANES as usize) * NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = (CUBE_DIM + PLANE_DIM - 1) / PLANE_DIM;

    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    let sub_part_size = PLANE_DIM * items_per_thread;
    let sub_part_start = block_start + plane_id * sub_part_size;

    // Initialize
    let total_hist_entries = (MAX_PLANES as usize) * NUM_BUCKETS;
    let entries_per_thread = (total_hist_entries + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..entries_per_thread {
        let idx = thread_id as usize + i * CUBE_DIM as usize;
        if idx < total_hist_entries {
            plane_hists[idx].store(0u32);
        }
    }
    sync_cube();

    // Phase 1: Load and WLMS
    let mut keys = Array::<u32>::new(items_per_thread as usize);
    let mut values = Array::<u32>::new(items_per_thread as usize);
    let mut offsets = Array::<u32>::new(items_per_thread as usize);
    let mut valid_flags = Array::<bool>::new(items_per_thread as usize);

    #[unroll]
    for i in 0..items_per_thread {
        let local_idx_in_subpart = lane_id + i * PLANE_DIM;
        let global_idx = sub_part_start + local_idx_in_subpart;
        let valid = global_idx < num_items;

        let key = select(valid, keys_in[global_idx as usize], 0u32);
        let value = select(valid, values_in[global_idx as usize], 0u32);
        keys[i as usize] = key;
        values[i as usize] = value;
        valid_flags[i as usize] = valid;

        let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

        let peer_mask = compute_peer_mask_valid(digit, valid, RADIX_BITS as u32);
        let rank_in_peers = count_lower_peers(peer_mask, lane_id);
        let total_peers = count_total_peers(peer_mask);
        let lowest_peer = find_lowest_peer(peer_mask);
        let is_leader = lane_id == lowest_peer && valid;

        let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
        let mut base_offset = 0u32;
        if is_leader {
            base_offset = plane_hists[hist_idx].fetch_add(total_peers);
        }
        base_offset = plane_shuffle(base_offset, lowest_peer);

        offsets[i as usize] = base_offset + rank_in_peers;
    }

    sync_cube();

    // Phase 2: Vertical prefix sum
    if thread_id < NUM_BUCKETS as u32 {
        let digit = thread_id as usize;
        let mut running_sum = 0u32;

        #[unroll]
        for p in 0..MAX_PLANES {
            if p < num_planes {
                let idx = (p as usize) * NUM_BUCKETS + digit;
                let count = plane_hists[idx].load();
                plane_hists[idx].store(running_sum);
                running_sum += count;
            }
        }
    }

    sync_cube();

    // Phase 3: Scatter
    #[unroll]
    for i in 0..items_per_thread {
        let valid = valid_flags[i as usize];

        if valid {
            let key = keys[i as usize];
            let value = values[i as usize];
            let offset_in_plane = offsets[i as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

            let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
            let plane_prefix = plane_hists[hist_idx].load();

            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];

            let final_pos = global_offset + plane_prefix + offset_in_plane;
            keys_out[final_pos as usize] = key;
            values_out[final_pos as usize] = value;
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute peer mask using WLMS algorithm, only including valid threads.
/// Returns a bitmask where bit i is set if lane i has the same digit AND is valid.
#[cube]
fn compute_peer_mask_valid(digit: u32, valid: bool, #[comptime] radix_bits: u32) -> Line<u32> {
    // Start with only valid lanes - invalid threads don't participate
    let mut peer_mask = plane_ballot(valid);

    // For each bit of the radix, filter out lanes that differ on that bit
    #[unroll]
    for k in 0..radix_bits {
        let bit_k = (digit >> k) & 1;
        let has_bit = bit_k != 0;
        // Get mask of all lanes where this bit is set
        let ballot = plane_ballot(has_bit);

        // XOR logic: if has_bit is true, we want lanes where ballot bit is also set
        // if has_bit is false, we want lanes where ballot bit is NOT set
        let xor_val = select(has_bit, 0u32, 0xFFFFFFFFu32);

        peer_mask[0] = peer_mask[0] & (ballot[0] ^ xor_val);
        if PLANE_DIM > 32 {
            peer_mask[1] = peer_mask[1] & (ballot[1] ^ xor_val);
        }
        if PLANE_DIM > 64 {
            peer_mask[2] = peer_mask[2] & (ballot[2] ^ xor_val);
        }
        if PLANE_DIM > 96 {
            peer_mask[3] = peer_mask[3] & (ballot[3] ^ xor_val);
        }
    }

    peer_mask
}

/// Count how many peers have a lower lane index (rank within peer group).
#[cube]
fn count_lower_peers(peer_mask: Line<u32>, lane_id: u32) -> u32 {
    let mut count = 0u32;

    // Word 0: lanes 0-31
    let lt_mask_0 = select(lane_id < 32, (1u32 << lane_id) - 1, 0xFFFFFFFFu32);
    count += (peer_mask[0] & lt_mask_0).count_ones();

    if PLANE_DIM > 32 {
        // Word 1: lanes 32-63
        let lt_mask_1 = select(
            lane_id < 32,
            0u32,
            select(lane_id < 64, (1u32 << (lane_id - 32)) - 1, 0xFFFFFFFFu32),
        );
        count += (peer_mask[1] & lt_mask_1).count_ones();
    }

    if PLANE_DIM > 64 {
        // Word 2: lanes 64-95
        let lt_mask_2 = select(
            lane_id < 64,
            0u32,
            select(lane_id < 96, (1u32 << (lane_id - 64)) - 1, 0xFFFFFFFFu32),
        );
        count += (peer_mask[2] & lt_mask_2).count_ones();
    }

    if PLANE_DIM > 96 {
        // Word 3: lanes 96-127
        let lt_mask_3 = select(lane_id < 96, 0u32, (1u32 << (lane_id - 96)) - 1);
        count += (peer_mask[3] & lt_mask_3).count_ones();
    }

    count
}

/// Count total peers.
#[cube]
fn count_total_peers(peer_mask: Line<u32>) -> u32 {
    let mut count = peer_mask[0].count_ones();
    if PLANE_DIM > 32 {
        count += peer_mask[1].count_ones();
    }
    if PLANE_DIM > 64 {
        count += peer_mask[2].count_ones();
    }
    if PLANE_DIM > 96 {
        count += peer_mask[3].count_ones();
    }
    count
}

/// Count trailing zeros in a u32.
#[cube]
fn count_trailing_zeros(x: u32) -> u32 {
    let is_zero = x == 0;
    // -x = ~x + 1, so x & -x isolates the lowest set bit
    let neg_x = (!x) + 1;
    let lowest_bit = x & neg_x;
    let below_lowest = lowest_bit - 1;
    let count = below_lowest.count_ones();
    select(is_zero, 32u32, count)
}

/// Find the lowest lane index among peers.
#[cube]
fn find_lowest_peer(peer_mask: Line<u32>) -> u32 {
    let mut result = 0u32;
    let mut found = false;

    if !found && peer_mask[0] != 0 {
        result = count_trailing_zeros(peer_mask[0]);
        found = true;
    }

    if PLANE_DIM > 32 && !found && peer_mask[1] != 0 {
        result = 32 + count_trailing_zeros(peer_mask[1]);
        found = true;
    }

    if PLANE_DIM > 64 && !found && peer_mask[2] != 0 {
        result = 64 + count_trailing_zeros(peer_mask[2]);
        found = true;
    }

    if PLANE_DIM > 96 && !found && peer_mask[3] != 0 {
        result = 96 + count_trailing_zeros(peer_mask[3]);
    }

    result
}
