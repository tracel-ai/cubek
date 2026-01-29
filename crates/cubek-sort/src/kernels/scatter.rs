//! Scatter kernel for radix sort.
//!
//! This kernel redistributes elements to their sorted positions based on
//! the computed global offsets.
//!
//! The parallel implementation uses Warp-Level Multi-Split (WLMS) to achieve
//! stable scatter with high parallelism.

use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use cubecl::prelude::*;

/// Compute peer mask using WLMS algorithm.
/// Returns a bitmask where bit i is set if lane i has the same digit as this lane.
///
/// The algorithm builds the mask iteratively: for each bit of the digit,
/// we compute ballot(bit_k) and XOR/AND appropriately to narrow down to exact matches.
#[cube]
fn compute_peer_mask(digit: u32, #[comptime] radix_bits: u32) -> Line<u32> {
    // Start with all lanes as potential peers
    let mut peer_mask = plane_ballot(true);

    // For each bit of the radix digit, narrow down to exact matches
    #[unroll]
    for k in 0..radix_bits {
        let bit_k = (digit >> k) & 1;
        let has_bit = bit_k != 0;
        let ballot = plane_ballot(has_bit);

        // If this lane has the bit set, keep only lanes that also have it set
        // If this lane doesn't have the bit, keep only lanes that also don't have it
        // peer_mask &= (has_bit ? ballot : ~ballot)
        // Compute XOR mask: 0 if has_bit, 0xFFFFFFFF if !has_bit
        let xor_val = select(has_bit, 0u32, 0xFFFFFFFFu32);

        peer_mask[0] = peer_mask[0] & (ballot[0] ^ xor_val);
        peer_mask[1] = peer_mask[1] & (ballot[1] ^ xor_val);
        peer_mask[2] = peer_mask[2] & (ballot[2] ^ xor_val);
        peer_mask[3] = peer_mask[3] & (ballot[3] ^ xor_val);
    }

    peer_mask
}

/// Count how many peers (lanes with same digit) have a lower lane index than this lane.
/// This gives the "local rank" within the peer group for stability.
#[cube]
fn count_lower_peers(peer_mask: Line<u32>) -> u32 {
    let lane_id = UNIT_POS_PLANE;

    // Create a mask of lanes with index < lane_id
    // We need (1 << lane_id) - 1 for the word containing lane_id, and all 1s for lower words

    let mut count = 0u32;

    // Word 0 (lanes 0-31)
    // If lane_id < 32: mask = (1 << lane_id) - 1
    // If lane_id >= 32: mask = 0xFFFFFFFF
    let lt_mask_0 = select(lane_id < 32, (1u32 << lane_id) - 1, 0xFFFFFFFFu32);
    count += (peer_mask[0] & lt_mask_0).count_ones();

    // Word 1 (lanes 32-63) - only if PLANE_DIM > 32
    // If lane_id < 32: mask = 0
    // If lane_id in [32, 64): mask = (1 << (lane_id - 32)) - 1
    // If lane_id >= 64: mask = 0xFFFFFFFF
    if PLANE_DIM > 32 {
        let in_word_1 = lane_id >= 32 && lane_id < 64;
        let lt_mask_1 = select(
            lane_id < 32,
            0u32,
            select(in_word_1, (1u32 << (lane_id - 32)) - 1, 0xFFFFFFFFu32),
        );
        count += (peer_mask[1] & lt_mask_1).count_ones();
    }

    // Word 2 (lanes 64-95)
    if PLANE_DIM > 64 {
        let in_word_2 = lane_id >= 64 && lane_id < 96;
        let lt_mask_2 = select(
            lane_id < 64,
            0u32,
            select(in_word_2, (1u32 << (lane_id - 64)) - 1, 0xFFFFFFFFu32),
        );
        count += (peer_mask[2] & lt_mask_2).count_ones();
    }

    // Word 3 (lanes 96-127)
    if PLANE_DIM > 96 {
        let lt_mask_3 = select(lane_id < 96, 0u32, (1u32 << (lane_id - 96)) - 1);
        count += (peer_mask[3] & lt_mask_3).count_ones();
    }

    count
}

/// Count total number of peers (lanes with same digit).
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

/// Count trailing zeros in a u32 using the formula: ctz(x) = 32 - popcount(x | -x) for x != 0
/// For x = 0, returns 32.
#[cube]
fn count_trailing_zeros(x: u32) -> u32 {
    // Formula: trailing_zeros(x) = popcount((x & -x) - 1) for x != 0
    // Alternatively: 31 - leading_zeros(x & -x) but we don't have leading_zeros either
    // Simplest: popcount of the mask of bits below the lowest set bit
    // If x = 0b...01000, then x-1 = 0b...00111, and popcount(x-1 & ~x) = trailing zeros
    // But simpler: (x & -x) isolates the lowest bit, then we need its position
    // Position = popcount((x & -x) - 1) = popcount of all bits below it

    // Handle zero case
    let is_zero = x == 0;
    // For x != 0: count = popcount((x & (-x)) - 1)
    // x & -x isolates the lowest set bit
    // Subtracting 1 gives all bits below it
    let neg_x = 0u32 - x; // Two's complement negation
    let lowest_bit = x & neg_x;
    let below_lowest = lowest_bit - 1;
    let count = below_lowest.count_ones();

    select(is_zero, 32u32, count)
}

/// Find the lowest lane index among peers.
#[cube]
fn find_lowest_peer(peer_mask: Line<u32>) -> u32 {
    // Find first set bit across all words (no early returns in cube functions)
    let mut result = 0u32;
    let mut found = false;

    // Check word 0
    if !found && peer_mask[0] != 0 {
        result = count_trailing_zeros(peer_mask[0]);
        found = true;
    }

    // Check word 1
    if PLANE_DIM > 32 && !found && peer_mask[1] != 0 {
        result = 32 + count_trailing_zeros(peer_mask[1]);
        found = true;
    }

    // Check word 2
    if PLANE_DIM > 64 && !found && peer_mask[2] != 0 {
        result = 64 + count_trailing_zeros(peer_mask[2]);
        found = true;
    }

    // Check word 3
    if PLANE_DIM > 96 && !found && peer_mask[3] != 0 {
        result = 96 + count_trailing_zeros(peer_mask[3]);
    }

    result
}

/// Scatter keys to their sorted positions for the current radix pass.
///
/// This kernel is STABLE - elements with the same digit maintain their
/// relative order from the input. This is critical for LSD radix sort.
///
/// Uses Warp-Level Multi-Split (WLMS) for parallel stable scatter:
/// 1. Each lane computes a peer mask (which lanes have same digit)
/// 2. Count peers with lower lane index = local rank within plane
/// 3. Lowest peer does atomic add to get base offset
/// 4. Broadcast base offset to all peers
/// 5. Final position = base_offset + local_rank
///
/// To maintain stability across planes within a block, planes process
/// sequentially using barriers. This ensures plane 0 updates counters
/// before plane 1, etc.
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
    // Atomic counters for block-level ranking
    let local_counters = SharedMemory::<Atomic<u32>>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    // PLANE_POS not available - compute manually
    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = (CUBE_DIM + PLANE_DIM - 1) / PLANE_DIM;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    // Initialize counters to zero (all threads help)
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            local_counters[bucket_idx].store(0u32);
        }
    }
    sync_cube();

    // Process elements in waves of CUBE_DIM elements
    #[unroll]
    for wave in 0..items_per_thread {
        let idx = block_start + wave * CUBE_DIM + thread_id;
        let valid = idx < num_items;

        // Load key (use 0 for invalid lanes - won't affect others)
        let key = select(valid, keys_in[idx as usize], 0u32);
        let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

        // Compute peer mask using WLMS (within this plane only)
        let peer_mask = compute_peer_mask(digit, RADIX_BITS as u32);

        // Count peers with lower lane index (for stability within plane)
        let local_rank_in_plane = count_lower_peers(peer_mask);

        // Count total peers in this plane
        let total_peers = count_total_peers(peer_mask);

        // Find lowest peer to do the atomic operation
        let lowest_peer = find_lowest_peer(peer_mask);
        let is_leader = lane_id == lowest_peer;

        // Process planes sequentially to maintain cross-plane stability
        // Each plane waits for all previous planes to finish
        let mut base_offset = 0u32;
        for p in 0..num_planes {
            if plane_id == p {
                // This plane's turn - leader does atomic add
                if valid && is_leader {
                    base_offset = local_counters[digit as usize].fetch_add(total_peers);
                }
                // Broadcast base offset from leader to all peers in this plane
                base_offset = plane_shuffle(base_offset, lowest_peer);
            }
            // All threads wait before next plane proceeds
            sync_cube();
        }

        // Compute final position and scatter
        if valid {
            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];
            let global_pos = global_offset + base_offset + local_rank_in_plane;
            keys_out[global_pos as usize] = key;
        }
    }
}

/// Scatter key-value pairs to their sorted positions (stable).
/// Uses WLMS algorithm for parallel stable scatter.
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
    // Atomic counters for block-level ranking
    let local_counters = SharedMemory::<Atomic<u32>>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    // PLANE_POS not available - compute manually
    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = (CUBE_DIM + PLANE_DIM - 1) / PLANE_DIM;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    // Initialize counters to zero (all threads help)
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            local_counters[bucket_idx].store(0u32);
        }
    }
    sync_cube();

    // Process elements in waves of CUBE_DIM elements
    #[unroll]
    for wave in 0..items_per_thread {
        let idx = block_start + wave * CUBE_DIM + thread_id;
        let valid = idx < num_items;

        // Load key and value (use 0 for invalid lanes)
        let key = select(valid, keys_in[idx as usize], 0u32);
        let value = select(valid, values_in[idx as usize], 0u32);
        let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

        // Compute peer mask using WLMS
        let peer_mask = compute_peer_mask(digit, RADIX_BITS as u32);

        // Count peers with lower lane index (for stability)
        let local_rank_in_plane = count_lower_peers(peer_mask);

        // Count total peers in this plane
        let total_peers = count_total_peers(peer_mask);

        // Find lowest peer to do the atomic operation
        let lowest_peer = find_lowest_peer(peer_mask);
        let is_leader = lane_id == lowest_peer;

        // Process planes sequentially to maintain cross-plane stability
        let mut base_offset = 0u32;
        for p in 0..num_planes {
            if plane_id == p {
                // This plane's turn - leader does atomic add
                if valid && is_leader {
                    base_offset = local_counters[digit as usize].fetch_add(total_peers);
                }
                // Broadcast base offset from leader to all peers in this plane
                base_offset = plane_shuffle(base_offset, lowest_peer);
            }
            // All threads wait before next plane proceeds
            sync_cube();
        }

        // Compute final position and scatter
        if valid {
            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];
            let global_pos = global_offset + base_offset + local_rank_in_plane;
            keys_out[global_pos as usize] = key;
            values_out[global_pos as usize] = value;
        }
    }
}
