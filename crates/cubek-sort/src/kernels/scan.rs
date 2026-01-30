use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

const MAX_WARPS: usize = 8;

/// Optimized scan kernel that computes prefix sums across all block histograms.
///
/// For each digit d, we need to compute:
/// - offsets[block, d] = sum of all histograms[b, d] for b < block, plus
///                       sum of all digit totals for digits < d
///
/// This version fuses the two sequential loops into one pass over the data.
#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    let mut digit_totals = SharedMemory::<u32>::new(NUM_BUCKETS);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = UNIT_POS_X / PLANE_DIM;

    // Phase 1: Each thread computes running exclusive prefix sum for its digit
    // and stores partial offsets (relative to digit 0)
    let mut running_sum = 0u32;
    if digit < NUM_BUCKETS as u32 {
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            let count = histograms[idx as usize];
            offsets[idx as usize] = running_sum;
            running_sum += count;
        }
        digit_totals[digit as usize] = running_sum;
    }
    sync_cube();

    // Phase 2: Compute exclusive prefix sum across all 256 digits
    // using two-level warp scan
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

    // Phase 3: Add warp prefix and update all offsets in one pass
    if digit < NUM_BUCKETS as u32 {
        let my_exclusive = digit_totals[digit as usize];
        let warp_prefix = warp_sums[warp_id as usize];
        let base_offset = warp_prefix + my_exclusive;

        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            offsets[idx as usize] += base_offset;
        }
    }
}
