use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use crate::components::key::SortKey;
use cubecl::prelude::*;

/// Histogram kernel that counts digit occurrences for each block.
///
/// Each block processes `items_per_thread * threads_per_block` keys and produces
/// a histogram of 256 digit counts. The histograms are stored contiguously in
/// global memory for subsequent prefix sum computation.
///
/// Uses warp-level ballot operations to reduce atomic contention: only the
/// leader thread (first thread with a given digit) performs the atomic add
/// for all threads in its warp with the same digit.
#[cube(launch_unchecked)]
pub fn histogram_kernel<K: SortKey>(
    keys: &Tensor<K>,
    histograms: &mut Tensor<u32>,
    num_items: u32,
    pass: u32,
    #[comptime] items_per_thread: u32,
) {
    let shared_hist = SharedMemory::<Atomic<u32>>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;
    let shift = pass * RADIX_BITS as u32;

    // Initialize shared histogram to zero
    #[allow(clippy::manual_div_ceil)]
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            shared_hist[bucket_idx].store(0u32);
        }
    }
    sync_cube();

    // Each thread handles items_per_thread keys total
    // Use warp-level ballot to reduce atomic contention
    #[unroll]
    for i in 0..items_per_thread {
        let idx = block_start + thread_id + i * CUBE_DIM;
        let valid = idx < num_items;

        // Load key and extract digit (use 0 for invalid to minimize divergence)
        let key = select(valid, K::to_radix(keys[idx as usize]), 0u32);
        let digit = (key >> shift) & DIGIT_MASK;

        // Use ballot to find all threads with the same digit
        let peer_mask = compute_peer_mask_hist(digit, valid);
        let count = count_set_bits_hist(peer_mask);
        let leader = find_first_set_bit_hist(peer_mask);

        // Only the leader performs the atomic add for the group
        if lane_id == leader && valid {
            shared_hist[digit as usize].fetch_add(count);
        }
    }
    sync_cube();

    // Write histogram to global memory
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            let global_idx = block_id as usize * NUM_BUCKETS + bucket_idx;
            histograms[global_idx] = shared_hist[bucket_idx].load();
        }
    }
}

#[cube]
fn compute_peer_mask_hist(digit: u32, valid: bool) -> Line<u32> {
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
fn count_set_bits_hist(mask: Line<u32>) -> u32 {
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
fn find_first_set_bit_hist(mask: Line<u32>) -> u32 {
    let mut result = 0u32;
    let mut found = false;

    if !found && mask[0] != 0 {
        result = count_trailing_zeros_hist(mask[0]);
        found = true;
    }

    if PLANE_DIM > 32 && !found && mask[1] != 0 {
        result = 32 + count_trailing_zeros_hist(mask[1]);
        found = true;
    }

    if PLANE_DIM > 64 && !found && mask[2] != 0 {
        result = 64 + count_trailing_zeros_hist(mask[2]);
        found = true;
    }

    if PLANE_DIM > 96 && !found && mask[3] != 0 {
        result = 96 + count_trailing_zeros_hist(mask[3]);
    }

    result
}

#[cube]
fn count_trailing_zeros_hist(x: u32) -> u32 {
    let is_zero = x == 0;
    let neg_x = (!x) + 1;
    let lowest_bit = x & neg_x;
    let count = (lowest_bit - 1).count_ones();
    select(is_zero, 32u32, count)
}
