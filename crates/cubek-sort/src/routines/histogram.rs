use crate::{
    key::{Radix, SortKey},
    routines::{NUM_BUCKETS, RADIX_BITS},
};

use super::warp_utils::{compute_peer_mask, count_set_bits, find_first_set_bit};
use cubecl::prelude::*;
use cubecl_std::tensor::layout::linear::LinearView;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistogramBlueprint {
    pub threads_per_block: u32,
    pub items_per_thread: u32,
}

/// Histogram kernel that counts digit occurrences for each block.
///
/// Each block processes `items_per_thread * threads_per_block` keys and produces
/// a histogram of 256 digit counts.
///
/// Uses warp-level ballot operations to reduce atomic contention: only the
/// leader thread (first thread with a given digit) performs the atomic add
/// for all threads in its warp with the same digit.
#[cube(launch_unchecked)]
pub fn histogram_kernel<K: SortKey<Radix = R>, R: Radix>(
    keys: &LinearView<K>,
    histograms: &mut Tensor<u32>,
    pass: u32,
    #[comptime] blueprint: HistogramBlueprint,
) {
    let shared_hist = SharedMemory::<Atomic<u32>>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let items_per_block = CUBE_DIM * blueprint.items_per_thread;
    let block_start = block_id * items_per_block;

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

    let num_keys = keys.shape() as u32;

    // Check if this is a full block (all items valid) - enables bounds check elimination
    let is_full_block = block_start + items_per_block <= num_keys;
    let zero_radix = R::cast_from(0u32);

    // Use warp-level ballot to reduce atomic contention
    // When is_full_block is true, the compiler can fold `valid` to true and eliminate branches
    for i in 0..blueprint.items_per_thread {
        let idx = block_start + thread_id + i * CUBE_DIM;
        let valid = is_full_block || idx < num_keys;

        // Clamp index to avoid out-of-bounds read (select doesn't short-circuit on GPU)
        let safe_idx = select(valid, idx, 0u32);
        let radix_key = select(valid, K::to_radix(keys[safe_idx as usize]), zero_radix);

        let shift_u32 = pass * RADIX_BITS as u32;
        let digit_radix = (radix_key >> R::cast_from(shift_u32)) & R::cast_from(0xFFu32);
        let digit = u32::cast_from(digit_radix);

        let peer_mask = compute_peer_mask(digit, valid);
        let count = count_set_bits(peer_mask);
        let leader = find_first_set_bit(peer_mask);

        if lane_id == leader {
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
