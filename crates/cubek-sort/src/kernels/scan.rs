use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

const MAX_WARPS: usize = 8;

#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    let mut digit_totals = SharedMemory::<u32>::new(NUM_BUCKETS);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = UNIT_POS_X / PLANE_DIM;

    if digit < NUM_BUCKETS as u32 {
        let mut running_sum = 0u32;
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            let count = histograms[idx as usize];
            offsets[idx as usize] = running_sum;
            running_sum += count;
        }
        digit_totals[digit as usize] = running_sum;
    }
    sync_cube();

    if digit < NUM_BUCKETS as u32 {
        let my_value = digit_totals[digit as usize];
        let my_exclusive = plane_exclusive_sum(my_value);
        digit_totals[digit as usize] = my_exclusive;

        if lane_id == PLANE_DIM - 1 {
            warp_sums[warp_id as usize] = my_exclusive + my_value;
        }
    }
    sync_cube();

    #[allow(clippy::manual_div_ceil)]
    let num_warps = (NUM_BUCKETS as u32 + PLANE_DIM - 1) / PLANE_DIM;
    if warp_id == 0 && lane_id < num_warps {
        let warp_total = warp_sums[lane_id as usize];
        let warp_prefix = plane_exclusive_sum(warp_total);
        warp_sums[lane_id as usize] = warp_prefix;
    }
    sync_cube();

    if digit < NUM_BUCKETS as u32 {
        let my_exclusive = digit_totals[digit as usize];
        let warp_prefix = warp_sums[warp_id as usize];
        digit_totals[digit as usize] = warp_prefix + my_exclusive;
    }
    sync_cube();

    if digit < NUM_BUCKETS as u32 {
        let base_offset = digit_totals[digit as usize];
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            offsets[idx as usize] += base_offset;
        }
    }
}
