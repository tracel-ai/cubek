//! Prefix scan kernel for radix sort.
//!
//! This kernel computes an exclusive prefix sum across all block histograms
//! to determine global write offsets for the scatter phase.

use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

/// Compute global offsets from per-block histograms.
///
/// This performs a global exclusive prefix sum across all histograms, treating
/// them as a single flattened array in column-major order (all blocks for digit 0,
/// then all blocks for digit 1, etc.).
///
/// The result tells each (block, digit) pair exactly where its first element
/// should go in the output array, accounting for:
/// 1. All elements from all blocks with smaller digit values
/// 2. All elements from previous blocks with the same digit value
///
/// This parallel implementation uses 256 threads (one per digit bucket).
/// Each thread:
/// 1. Computes prefix sum across all blocks for its digit
/// 2. Computes the total count for its digit
/// 3. Uses a workgroup prefix sum to get the base offset for its digit
/// 4. Adds the base offset to all positions
///
/// # Arguments
///
/// * `histograms` - Input histograms of shape [num_blocks, NUM_BUCKETS] (row-major)
/// * `offsets` - Output offsets of shape [num_blocks, NUM_BUCKETS] (row-major)
/// * `num_blocks` - Number of thread blocks used in histogram kernel
#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    // Shared memory for digit totals (for cross-digit prefix sum)
    let mut digit_totals = SharedMemory::<u32>::new(NUM_BUCKETS);

    let digit = UNIT_POS_X;

    if digit < NUM_BUCKETS as u32 {
        // Phase 1: Each thread computes exclusive prefix sum for its digit across all blocks
        // and stores intermediate results + computes total
        let mut running_sum = 0u32;

        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            let count = histograms[idx as usize];
            // Store the running sum (this is the intra-digit prefix)
            offsets[idx as usize] = running_sum;
            running_sum += count;
        }

        // Store total count for this digit
        digit_totals[digit as usize] = running_sum;
    }
    sync_cube();

    // Phase 2: Compute exclusive prefix sum across digit totals
    // This gives us the base offset for each digit
    // We use a simple sequential scan on thread 0 since NUM_BUCKETS is small (256)
    if UNIT_POS_X == 0 {
        let mut prefix = 0u32;
        for d in 0..NUM_BUCKETS {
            let total = digit_totals[d];
            digit_totals[d] = prefix;
            prefix += total;
        }
    }
    sync_cube();

    // Phase 3: Each thread adds its digit's base offset to all its positions
    if digit < NUM_BUCKETS as u32 {
        let base_offset = digit_totals[digit as usize];

        for block in 0..num_blocks {
            let idx = block * NUM_BUCKETS as u32 + digit;
            offsets[idx as usize] = offsets[idx as usize] + base_offset;
        }
    }
}
