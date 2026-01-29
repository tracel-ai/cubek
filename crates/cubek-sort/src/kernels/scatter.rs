//! Scatter kernel for radix sort.
//!
//! This kernel redistributes elements to their sorted positions based on
//! the computed global offsets.

use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use cubecl::prelude::*;

/// Scatter keys to their sorted positions for the current radix pass.
///
/// This kernel is STABLE - elements with the same digit maintain their
/// relative order from the input. This is critical for LSD radix sort.
///
/// To achieve stability, we process elements in strict index order using
/// a single thread per block (thread 0). This is slower but correct.
/// A more optimized version would use warp-level primitives.
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
    // Local counters for ranking within this block
    let mut local_counters = SharedMemory::<u32>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;
    let block_end_raw = block_start + items_per_block;
    let block_end = if block_end_raw < num_items {
        block_end_raw
    } else {
        num_items
    };

    // Initialize counters to zero (all threads help)
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            local_counters[bucket_idx] = 0u32;
        }
    }
    sync_cube();

    // Thread 0 processes all elements in order to maintain stability
    if thread_id == 0 {
        for idx in block_start..block_end {
            let key = keys_in[idx as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

            // Get and increment local rank
            let local_rank = local_counters[digit as usize];
            local_counters[digit as usize] = local_rank + 1;

            // Get the global offset for this (block, digit) pair
            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];

            // Final position = global offset + local rank
            let global_pos = global_offset + local_rank;

            keys_out[global_pos as usize] = key;
        }
    }
}

/// Scatter key-value pairs to their sorted positions (stable).
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
    // Local counters for ranking within this block
    let mut local_counters = SharedMemory::<u32>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;
    let block_end_raw = block_start + items_per_block;
    let block_end = if block_end_raw < num_items {
        block_end_raw
    } else {
        num_items
    };

    // Initialize counters to zero (all threads help)
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            local_counters[bucket_idx] = 0u32;
        }
    }
    sync_cube();

    // Thread 0 processes all elements in order to maintain stability
    if thread_id == 0 {
        for idx in block_start..block_end {
            let key = keys_in[idx as usize];
            let value = values_in[idx as usize];
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;

            // Get and increment local rank
            let local_rank = local_counters[digit as usize];
            local_counters[digit as usize] = local_rank + 1;

            // Get the global offset for this (block, digit) pair
            let block_offset_idx = block_id as usize * NUM_BUCKETS + digit as usize;
            let global_offset = block_offsets[block_offset_idx];

            // Final position = global offset + local rank
            let global_pos = global_offset + local_rank;

            keys_out[global_pos as usize] = key;
            values_out[global_pos as usize] = value;
        }
    }
}
