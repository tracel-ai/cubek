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
/// For simplicity, this uses a single-threaded approach since the histogram
/// is relatively small (num_blocks * 256 elements).
///
/// # Arguments
///
/// * `histograms` - Input histograms of shape [num_blocks, NUM_BUCKETS] (row-major)
/// * `offsets` - Output offsets of shape [num_blocks, NUM_BUCKETS] (row-major)
/// * `num_blocks` - Number of thread blocks used in histogram kernel
#[cube(launch_unchecked)]
pub fn scan_kernel(histograms: &Tensor<u32>, offsets: &mut Tensor<u32>, num_blocks: u32) {
    // This kernel is launched with a single thread
    // We need to compute a global prefix sum where elements are ordered:
    // - First all (block, digit=0) entries
    // - Then all (block, digit=1) entries
    // - etc.
    //
    // But histograms are stored as [block][digit] (row-major).
    // So we iterate in column-major order: for each digit, for each block.

    if UNIT_POS_X == 0 {
        let mut running_sum = 0u32;

        // Process in column-major order (digit, then block)
        // This ensures all digit=0 elements come before digit=1 elements
        for digit in 0..NUM_BUCKETS as u32 {
            for block in 0..num_blocks {
                let idx = block * NUM_BUCKETS as u32 + digit;
                let count = histograms[idx as usize];
                offsets[idx as usize] = running_sum;
                running_sum += count;
            }
        }
    }
}
