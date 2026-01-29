use crate::components::config::{DIGIT_MASK, NUM_BUCKETS, RADIX_BITS};
use crate::components::key::SortKey;
use cubecl::prelude::*;

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
    let items_per_block = CUBE_DIM * items_per_thread;
    let block_start = block_id * items_per_block;

    #[allow(clippy::manual_div_ceil)]
    let buckets_per_thread = (NUM_BUCKETS + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            shared_hist[bucket_idx].store(0u32);
        }
    }
    sync_cube();

    #[unroll]
    for i in 0..items_per_thread {
        let idx = block_start + thread_id + i * CUBE_DIM;
        if idx < num_items {
            let key = K::to_radix(keys[idx as usize]);
            let digit = (key >> (pass * RADIX_BITS as u32)) & DIGIT_MASK;
            shared_hist[digit as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    for i in 0..buckets_per_thread {
        let bucket_idx = thread_id as usize + i * CUBE_DIM as usize;
        if bucket_idx < NUM_BUCKETS {
            let global_idx = block_id as usize * NUM_BUCKETS + bucket_idx;
            histograms[global_idx] = shared_hist[bucket_idx].load();
        }
    }
}
