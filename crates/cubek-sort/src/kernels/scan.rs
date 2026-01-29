//! Scan kernel for radix sort.
//!
//! Computes global write offsets by performing an exclusive prefix sum across
//! all block histograms. The result tells each (block, digit) pair exactly
//! where its first element should be written in the output array.
//!
//! Uses a two-level parallel scan approach:
//! 1. Each thread computes per-digit totals across blocks
//! 2. Parallel warp-level exclusive scan using plane_exclusive_sum
//! 3. Cross-warp exclusive scan to combine warp totals
//! 4. Final combination to produce global exclusive prefix sums

use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

/// Maximum number of warps (256 threads / 32 = 8 warps)
const MAX_WARPS: usize = 8;

/// Compute global offsets from per-block histograms.
#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    let mut digit_totals = SharedMemory::<u32>::new(NUM_BUCKETS);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = UNIT_POS_X / PLANE_DIM;

    // Phase 1: Each thread computes exclusive prefix sum for its digit across all blocks
    // This gives us per-block offsets within each digit, plus the total count per digit
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

    // Phase 2: Two-level parallel exclusive prefix sum across digit totals
    // This computes the base offset for each digit (sum of all preceding digits)

    // Step 2a: Exclusive warp scan using built-in plane_exclusive_sum
    // Each warp computes an exclusive prefix sum of its elements
    if digit < NUM_BUCKETS as u32 {
        let my_value = digit_totals[digit as usize];

        // plane_exclusive_sum gives us sum of all values to the left (excluding self)
        let my_exclusive = plane_exclusive_sum(my_value);

        // Store exclusive scan result
        digit_totals[digit as usize] = my_exclusive;

        // Last lane stores the warp's total (exclusive sum + own value = inclusive sum)
        let last_lane = PLANE_DIM - 1;
        if lane_id == last_lane {
            warp_sums[warp_id as usize] = my_exclusive + my_value;
        }
    }
    sync_cube();

    // Step 2b: Exclusive scan of warp sums (only first warp participates)
    // This computes the prefix for each warp (sum of all preceding warps)
    #[allow(clippy::manual_div_ceil)]
    let num_warps = (NUM_BUCKETS as u32 + PLANE_DIM - 1) / PLANE_DIM;
    if warp_id == 0 && lane_id < num_warps {
        let warp_total = warp_sums[lane_id as usize];

        // Exclusive scan of warp totals
        let warp_prefix = plane_exclusive_sum(warp_total);
        warp_sums[lane_id as usize] = warp_prefix;
    }
    sync_cube();

    // Step 2c: Add warp prefix to each thread's exclusive scan result
    // Final result: exclusive prefix sum across all 256 digits
    if digit < NUM_BUCKETS as u32 {
        let my_exclusive = digit_totals[digit as usize];
        let warp_prefix = warp_sums[warp_id as usize];
        digit_totals[digit as usize] = warp_prefix + my_exclusive;
    }
    sync_cube();

    // Phase 3: Add digit base offset to all positions
    // Each (block, digit) position gets the digit's base offset added
    if digit < NUM_BUCKETS as u32 {
        let base_offset = digit_totals[digit as usize];
        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            offsets[idx as usize] = offsets[idx as usize] + base_offset;
        }
    }
}
