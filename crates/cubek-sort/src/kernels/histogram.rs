//! Histogram kernel for radix sort.
//!
//! This kernel counts the occurrences of each digit value within each thread block.
//! The output is a 2D histogram of shape [num_blocks, NUM_BUCKETS].

use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use cubecl::prelude::*;

/// Compute per-block histograms of digit values.
///
/// Each thread block processes a contiguous chunk of keys and counts how many
/// keys have each digit value (0-255) for the current radix pass.
///
/// This implementation uses atomic operations on shared memory for counting,
/// which is efficient and allows all threads to contribute in parallel.
#[cube(launch_unchecked)]
pub fn histogram_kernel(
    keys: &Tensor<u32>,
    histograms: &mut Tensor<u32>,
    num_items: u32,
    pass: u32,
    #[comptime] items_per_thread: u32,
    #[comptime] _num_buckets: u32,
    #[comptime] _cube_size: u32,
) {
    // Shared memory histogram with atomic access
    let shared_hist = SharedMemory::<Atomic<u32>>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    // Initialize shared histogram to zero
    // Each thread initializes multiple buckets if needed
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            shared_hist[bucket_idx].store(0u32);
        }
    }
    sync_cube();

    // Each thread processes its assigned elements and atomically increments counts
    #[unroll]
    for i in 0..items_per_thread {
        let idx = block_start + thread_id + i * CUBE_DIM;
        if idx < num_items {
            let key = keys[idx as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;
            shared_hist[digit as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    // Write shared histogram to global memory
    // Each thread writes multiple buckets if needed
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            let global_idx = block_id as usize * NUM_BUCKETS + bucket_idx;
            histograms[global_idx] = shared_hist[bucket_idx].load();
        }
    }
}
