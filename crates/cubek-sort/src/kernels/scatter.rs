use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use crate::components::key::SortKey;
use cubecl::prelude::*;

const NUM_BUCKETS_U32: u32 = NUM_BUCKETS as u32;

#[cube(launch_unchecked)]
pub fn scatter_kernel<KIn: SortKey, KOut: SortKey>(
    keys_in: &Tensor<KIn>,
    keys_out: &mut Tensor<KOut>,
    values_in: &Tensor<u32>,
    values_out: &mut Tensor<u32>,
    block_offsets: &Tensor<u32>,
    num_items: u32,
    pass: u32,
    #[comptime] items_per_thread: u32,
    #[comptime] has_values: bool,
    #[comptime] num_planes: u32,
    #[comptime] items_per_block: u32,
) {
    // Plane histograms for warp-level ranking
    let plane_hists = SharedMemory::<Atomic<u32>>::new((num_planes as usize) * NUM_BUCKETS);
    // Shared memory buffer for keys (reused for local reordering)
    let mut shared_keys = SharedMemory::<u32>::new(items_per_block as usize);
    // Shared memory buffer for values
    let mut shared_values = SharedMemory::<u32>::new(items_per_block as usize);
    // Where each digit starts in shared memory (exclusive prefix sum of digit counts)
    let mut digit_start = SharedMemory::<u32>::new(NUM_BUCKETS);
    // Global write offset for each digit (block_offset - digit_start, so global_pos = this + local_idx)
    let mut digit_global = SharedMemory::<u32>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let plane_id = UNIT_POS / PLANE_DIM;

    let block_start = block_id * items_per_block;
    let sub_part_size = PLANE_DIM * items_per_thread;
    let sub_part_start = block_start + plane_id * sub_part_size;

    // Initialize plane histograms
    let total_hist_entries = (num_planes as usize) * NUM_BUCKETS;
    #[allow(clippy::manual_div_ceil)]
    let entries_per_thread = (total_hist_entries + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..entries_per_thread {
        let idx = thread_id as usize + i * CUBE_DIM as usize;
        if idx < total_hist_entries {
            plane_hists[idx].store(0u32);
        }
    }
    sync_cube();

    // Register arrays for per-thread data
    let mut keys = Array::<u32>::new(items_per_thread as usize);
    let mut digits = Array::<u32>::new(items_per_thread as usize); // Store digits to avoid recomputation
    let mut values = Array::<u32>::new(items_per_thread as usize);
    let mut local_offsets = Array::<u32>::new(items_per_thread as usize);
    let mut valid_flags = Array::<bool>::new(items_per_thread as usize);

    let shift = pass * RADIX_BITS as u32;

    // Phase 1: Load keys and compute warp-level ranking
    #[unroll]
    for i in 0..items_per_thread {
        let local_idx = lane_id + i * PLANE_DIM;
        let global_idx = sub_part_start + local_idx;
        let valid = global_idx < num_items;

        let raw_key = select(valid, keys_in[global_idx as usize], KIn::from_radix(0u32));
        let key = KIn::to_radix(raw_key);
        keys[i as usize] = key;
        valid_flags[i as usize] = valid;

        if has_values {
            values[i as usize] = select(valid, values_in[global_idx as usize], 0u32);
        }

        let digit = (key >> shift) & DIGIT_MASK;
        digits[i as usize] = digit; // Store for later use

        // Warp-level ranking using ballot
        let peer_mask = compute_peer_mask(digit, valid);
        let rank = count_lower_peers(peer_mask, lane_id);
        let total = count_set_bits(peer_mask);
        let leader = find_first_set_bit(peer_mask);
        let is_leader = lane_id == leader && valid;

        // Atomically add to plane histogram, get base offset
        let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
        let mut base = 0u32;
        if is_leader {
            base = plane_hists[hist_idx].fetch_add(total);
        }
        base = plane_shuffle(base, leader);

        local_offsets[i as usize] = base + rank;
    }
    sync_cube();

    // Phase 2: Reduce plane histograms and compute digit starts
    // Each thread handles one digit (first 256 threads)
    if thread_id < NUM_BUCKETS_U32 {
        let mut sum = 0u32;

        // Sum across all planes for this digit, converting to exclusive prefix within each plane
        #[unroll]
        for p in 0..num_planes {
            let idx = (p as usize) * NUM_BUCKETS + thread_id as usize;
            let count = plane_hists[idx].load();
            plane_hists[idx].store(sum);
            sum += count;
        }

        // sum is now total count for this digit in this block
        // Store it temporarily in digit_start for the prefix sum
        digit_start[thread_id as usize] = sum;
    }
    sync_cube();

    // Compute exclusive prefix sum across 256 digits using warp-level primitives
    // Optimized to use fewer sync barriers by computing warp totals inline
    #[allow(clippy::manual_div_ceil)]
    let num_digit_warps = (NUM_BUCKETS_U32 + PLANE_DIM - 1) / PLANE_DIM;

    // Step 1: Warp-level exclusive scan and store warp totals
    if thread_id < NUM_BUCKETS_U32 {
        let val = digit_start[thread_id as usize];
        let warp_exclusive = plane_exclusive_sum(val);
        let my_inclusive = warp_exclusive + val;
        let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

        digit_start[thread_id as usize] = warp_exclusive;

        let digit_warp_id = thread_id / PLANE_DIM;
        if lane_id == 0 {
            digit_global[digit_warp_id as usize] = warp_total;
        }
    }
    sync_cube();

    // Step 2: First warp computes prefix sum of warp totals, then all threads
    // read their warp prefix and compute final offset in one pass
    if thread_id < num_digit_warps {
        let warp_total = digit_global[thread_id as usize];
        let warp_prefix = plane_exclusive_sum(warp_total);
        digit_global[thread_id as usize] = warp_prefix;
    }
    sync_cube();

    // Step 3: Add warp prefix and compute global offset
    if thread_id < NUM_BUCKETS_U32 {
        let digit_warp_id = thread_id / PLANE_DIM;
        let warp_prefix = digit_global[digit_warp_id as usize];
        let my_start = digit_start[thread_id as usize] + warp_prefix;
        digit_start[thread_id as usize] = my_start;

        let block_offset = block_offsets[block_id as usize * NUM_BUCKETS + thread_id as usize];
        digit_global[thread_id as usize] = block_offset - my_start;
    }
    sync_cube();

    // Phase 3: Scatter keys to shared memory (local reordering by digit)
    #[unroll]
    for i in 0..items_per_thread {
        if valid_flags[i as usize] {
            let key = keys[i as usize];
            let digit = digits[i as usize]; // Use stored digit instead of recomputing
            let offset_in_plane = local_offsets[i as usize];

            let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
            let plane_prefix = plane_hists[hist_idx].load();
            let digit_base = digit_start[digit as usize];
            let local_pos = digit_base + plane_prefix + offset_in_plane;

            shared_keys[local_pos as usize] = key;
            if has_values {
                shared_values[local_pos as usize] = values[i as usize];
            }
        }
    }
    sync_cube();

    // Phase 4: Coalesced read from shared memory and write to global
    // Keys are now grouped by digit, so sequential reads are cache-friendly
    // and we can compute global position directly
    let items_in_block = select(
        block_start + items_per_block <= num_items,
        items_per_block,
        select(num_items > block_start, num_items - block_start, 0u32),
    );

    #[unroll]
    for i in 0..items_per_thread {
        let local_idx = thread_id + i * CUBE_DIM;
        if local_idx < items_in_block {
            let key = shared_keys[local_idx as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

            // global_pos = block_offset[digit] - digit_start[digit] + local_idx
            //            = digit_global[digit] + local_idx
            let global_pos = digit_global[digit as usize] + local_idx;

            keys_out[global_pos as usize] = KOut::from_radix(key);

            if has_values {
                values_out[global_pos as usize] = shared_values[local_idx as usize];
            }
        }
    }
}

#[cube]
fn compute_peer_mask(digit: u32, valid: bool) -> Line<u32> {
    let mut mask = plane_ballot(valid);

    #[unroll]
    for k in 0..RADIX_BITS {
        let has_bit = ((digit >> k) & 1) != 0;
        let ballot = plane_ballot(has_bit);
        let xor_val = select(has_bit, 0u32, 0xFFFFFFFFu32);

        mask[0] &= ballot[0] ^ xor_val;
        if PLANE_DIM > 32 {
            mask[1] &= ballot[1] ^ xor_val;
        }
        if PLANE_DIM > 64 {
            mask[2] &= ballot[2] ^ xor_val;
        }
        if PLANE_DIM > 96 {
            mask[3] &= ballot[3] ^ xor_val;
        }
    }
    mask
}

#[cube]
fn count_lower_peers(mask: Line<u32>, lane_id: u32) -> u32 {
    let mut count = 0u32;

    let lt_mask_0 = select(lane_id < 32, (1u32 << lane_id) - 1, 0xFFFFFFFFu32);
    count += (mask[0] & lt_mask_0).count_ones();

    if PLANE_DIM > 32 {
        let lt_mask_1 = select(
            lane_id < 32,
            0u32,
            select(lane_id < 64, (1u32 << (lane_id - 32)) - 1, 0xFFFFFFFFu32),
        );
        count += (mask[1] & lt_mask_1).count_ones();
    }

    if PLANE_DIM > 64 {
        let lt_mask_2 = select(
            lane_id < 64,
            0u32,
            select(lane_id < 96, (1u32 << (lane_id - 64)) - 1, 0xFFFFFFFFu32),
        );
        count += (mask[2] & lt_mask_2).count_ones();
    }

    if PLANE_DIM > 96 {
        let lt_mask_3 = select(lane_id < 96, 0u32, (1u32 << (lane_id - 96)) - 1);
        count += (mask[3] & lt_mask_3).count_ones();
    }

    count
}

#[cube]
fn count_set_bits(mask: Line<u32>) -> u32 {
    let mut count = mask[0].count_ones();
    if PLANE_DIM > 32 {
        count += mask[1].count_ones();
    }
    if PLANE_DIM > 64 {
        count += mask[2].count_ones();
    }
    if PLANE_DIM > 96 {
        count += mask[3].count_ones();
    }
    count
}

#[cube]
fn find_first_set_bit(mask: Line<u32>) -> u32 {
    let mut result = 0u32;
    let mut found = false;

    if !found && mask[0] != 0 {
        result = count_trailing_zeros(mask[0]);
        found = true;
    }

    if PLANE_DIM > 32 && !found && mask[1] != 0 {
        result = 32 + count_trailing_zeros(mask[1]);
        found = true;
    }

    if PLANE_DIM > 64 && !found && mask[2] != 0 {
        result = 64 + count_trailing_zeros(mask[2]);
        found = true;
    }

    if PLANE_DIM > 96 && !found && mask[3] != 0 {
        result = 96 + count_trailing_zeros(mask[3]);
    }

    result
}

#[cube]
fn count_trailing_zeros(x: u32) -> u32 {
    let is_zero = x == 0;
    let neg_x = (!x) + 1;
    let lowest_bit = x & neg_x;
    let count = (lowest_bit - 1).count_ones();
    select(is_zero, 32u32, count)
}
