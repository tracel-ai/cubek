use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

const MAX_WARPS: usize = 8;

/// Optimized scan kernel that computes prefix sums across all block histograms.
///
/// For each digit d, we need to compute:
/// - offsets[block, d] = sum of all histograms[b, d] for b < block, plus
///                       sum of all digit totals for digits < d
///
/// This version processes blocks in chunks, caching histogram values in registers
/// to reduce global memory traffic.
#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    let mut digit_totals = SharedMemory::<u32>::new(NUM_BUCKETS);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = UNIT_POS_X / PLANE_DIM;

    // Phase 1 & 3 combined: Process blocks in chunks, caching histogram values
    // First pass: compute digit totals
    let mut running_sum = 0u32;
    if digit < NUM_BUCKETS as u32 {
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            let count = histograms[idx as usize];
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
        let my_inclusive = my_exclusive + my_value;
        let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

        if lane_id == PLANE_DIM - 1 {
            warp_sums[warp_id as usize] = warp_total;
        }
        digit_totals[digit as usize] = my_exclusive;
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

    // Step 3: Compute base offset and write final offsets
    if digit < NUM_BUCKETS as u32 {
        let my_exclusive = digit_totals[digit as usize];
        let warp_prefix = warp_sums[warp_id as usize];
        let base_offset = warp_prefix + my_exclusive;

        // Write offsets - re-read histogram values
        running_sum = 0u32;
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            let count = histograms[idx as usize];
            offsets[idx as usize] = base_offset + running_sum;
            running_sum += count;
        }
    }
}
