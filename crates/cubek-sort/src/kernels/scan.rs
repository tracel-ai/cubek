use crate::components::config::NUM_BUCKETS;
use cubecl::prelude::*;

// Maximum warps per block. Must accommodate scan_dim / PLANE_DIM.
// PLANE_DIM can be as low as 8 on Intel, so with scan_dim=256 we need up to 32 warps.
const MAX_WARPS: usize = 32;

/// Phase A: Compute digit totals by summing histogram values across all blocks.
/// Launches 256 blocks (one per digit), each with SCAN_DIM threads that
/// cooperatively sum all histogram blocks for that digit.
#[cube(launch_unchecked)]
pub fn scan_sum_kernel(
    histograms: &Tensor<u32>,
    digit_totals: &mut Tensor<u32>,
    num_hist_blocks: u32,
    #[comptime] scan_dim: u32,
) {
    let mut shared_sum = SharedMemory::<u32>::new(scan_dim as usize);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = CUBE_POS_X; // Which digit this block handles (0-255)
    let tid = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = PLANE_POS;

    // Initialize shared memory to avoid reading uninitialized values
    // when num_warps > PLANE_DIM (e.g., Intel with PLANE_DIM=8)
    if tid < MAX_WARPS as u32 {
        warp_sums[tid as usize] = 0u32;
        shared_sum[tid as usize] = 0u32;
    }
    sync_cube();

    // Each thread sums its strided portion of histogram blocks
    let mut my_sum = 0u32;
    let mut block_idx = tid;
    for _ in 0..num_hist_blocks.div_ceil(scan_dim) {
        if block_idx < num_hist_blocks {
            let hist_idx = block_idx * NUM_BUCKETS as u32 + digit;
            my_sum += histograms[hist_idx as usize];
        }
        block_idx += scan_dim;
    }

    // Reduce within warp
    let warp_sum = plane_sum(my_sum);
    if lane_id == 0 {
        warp_sums[warp_id as usize] = warp_sum;
    }
    sync_cube();

    // Reduce across warps (first warp does this)
    #[allow(clippy::manual_div_ceil)]
    let num_warps = (scan_dim + PLANE_DIM - 1) / PLANE_DIM;
    if warp_id == 0 && lane_id < num_warps {
        shared_sum[lane_id as usize] = warp_sums[lane_id as usize];
    }
    sync_cube();

    // Thread 0 sums all warp totals and writes result
    if tid == 0 {
        let mut total = 0u32;
        for w in 0..num_warps {
            total += shared_sum[w as usize];
        }
        digit_totals[digit as usize] = total;
    }
}

/// Phase B: Compute exclusive prefix sum across digit totals.
/// Single block with 256 threads.
#[cube(launch_unchecked)]
pub fn scan_prefix_totals_kernel(digit_totals: &Tensor<u32>, digit_prefixes: &mut Tensor<u32>) {
    let mut shared = SharedMemory::<u32>::new(NUM_BUCKETS);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = PLANE_POS;

    // Initialize warp_sums to avoid reading uninitialized values
    if digit < MAX_WARPS as u32 {
        warp_sums[digit as usize] = 0u32;
    }
    sync_cube();

    let my_value = if digit < NUM_BUCKETS as u32 {
        digit_totals[digit as usize]
    } else {
        #[allow(clippy::useless_conversion)]
        0u32.into()
    };

    // Intra-warp prefix sum
    let my_exclusive = plane_exclusive_sum(my_value);
    let my_inclusive = my_exclusive + my_value;
    let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

    if lane_id == PLANE_DIM - 1 {
        warp_sums[warp_id as usize] = warp_total;
    }
    shared[digit as usize] = my_exclusive;
    sync_cube();

    // Inter-warp prefix sum
    // All threads must participate in plane ops.
    #[allow(clippy::manual_div_ceil)]
    let num_warps = (NUM_BUCKETS as u32 + PLANE_DIM - 1) / PLANE_DIM;
    let warp_input = if warp_id == 0 && lane_id < num_warps {
        warp_sums[lane_id as usize]
    } else {
        #[allow(clippy::useless_conversion)]
        0u32.into()
    };
    let warp_prefix = plane_exclusive_sum(warp_input);
    if warp_id == 0 && lane_id < num_warps {
        warp_sums[lane_id as usize] = warp_prefix;
    }
    sync_cube();

    // Combine and write
    if digit < NUM_BUCKETS as u32 {
        let warp_prefix = warp_sums[warp_id as usize];
        digit_prefixes[digit as usize] = shared[digit as usize] + warp_prefix;
    }
}

/// Phase C: Compute within-digit offsets using cooperative prefix + base offsets.
/// Launches 256 blocks (one per digit), each with SCAN_DIM threads.
#[cube(launch_unchecked)]
pub fn scan_offsets(
    histograms: &Tensor<u32>,
    digit_prefixes: &Tensor<u32>,
    offsets: &mut Tensor<u32>,
    num_hist_blocks: u32,
    #[comptime] scan_dim: u32,
) {
    let mut g_scan = SharedMemory::<u32>::new(scan_dim as usize);
    let mut warp_sums = SharedMemory::<u32>::new(MAX_WARPS);

    let digit = CUBE_POS_X;
    let tid = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let warp_id = PLANE_POS;

    // Load base offset for this digit (from cross-digit prefix sum)
    let base_offset = digit_prefixes[digit as usize];

    let full_chunks = num_hist_blocks / scan_dim;
    let mut reduction = 0u32;

    // Process full chunks
    for chunk in 0..full_chunks {
        let block_idx = chunk * scan_dim + tid;
        let hist_idx = block_idx * NUM_BUCKETS as u32 + digit;
        let my_value = histograms[hist_idx as usize];
        g_scan[tid as usize] = my_value;

        // Intra-warp exclusive prefix
        let my_exclusive = plane_exclusive_sum(my_value);
        let my_inclusive = my_exclusive + my_value;
        let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

        if lane_id == PLANE_DIM - 1 {
            warp_sums[warp_id as usize] = warp_total;
        }
        g_scan[tid as usize] = my_exclusive;
        sync_cube();

        // Inter-warp prefix (all threads participate; partial participation is UB)
        #[allow(clippy::manual_div_ceil)]
        let num_warps = (scan_dim + PLANE_DIM - 1) / PLANE_DIM;
        let warp_input = if warp_id == 0 && lane_id < num_warps {
            warp_sums[lane_id as usize]
        } else {
            #[allow(clippy::useless_conversion)]
            0u32.into()
        };
        let warp_prefix = plane_exclusive_sum(warp_input);
        if warp_id == 0 && lane_id < num_warps {
            warp_sums[lane_id as usize] = warp_prefix;
        }
        sync_cube();

        // Combine and write offset
        let warp_prefix = warp_sums[warp_id as usize];
        let final_exclusive = g_scan[tid as usize] + warp_prefix;
        let out_idx = block_idx * NUM_BUCKETS as u32 + digit;
        offsets[out_idx as usize] = base_offset + reduction + final_exclusive;

        // Update reduction for next chunk
        if tid == scan_dim - 1 {
            reduction += final_exclusive + my_value;
        }
        sync_cube();
        if tid == scan_dim - 1 {
            g_scan[0] = reduction;
        }
        sync_cube();
        reduction = g_scan[0];
        sync_cube();
    }

    // Process partial chunk
    let partial_start = full_chunks * scan_dim;
    let remaining = num_hist_blocks - partial_start;

    if remaining > 0 {
        let block_idx = partial_start + tid;
        let hist_idx = block_idx * NUM_BUCKETS as u32 + digit;
        // Clamp index to avoid out-of-bounds read (select doesn't short-circuit on GPU)

        let my_value = if tid < remaining {
            histograms[hist_idx as usize]
        } else {
            #[allow(clippy::useless_conversion)]
            0u32.into()
        };

        let my_exclusive = plane_exclusive_sum(my_value);
        let my_inclusive = my_exclusive + my_value;
        let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

        if lane_id == PLANE_DIM - 1 {
            warp_sums[warp_id as usize] = warp_total;
        }
        g_scan[tid as usize] = my_exclusive;
        sync_cube();

        // All threads must participate in plane ops (partial participation is UB)
        #[allow(clippy::manual_div_ceil)]
        let num_warps = (scan_dim + PLANE_DIM - 1) / PLANE_DIM;
        let warp_input = if warp_id == 0 && lane_id < num_warps {
            warp_sums[lane_id as usize]
        } else {
            #[allow(clippy::useless_conversion)]
            0u32.into()
        };
        let warp_prefix = plane_exclusive_sum(warp_input);
        if warp_id == 0 && lane_id < num_warps {
            warp_sums[lane_id as usize] = warp_prefix;
        }
        sync_cube();

        if tid < remaining {
            let warp_prefix = warp_sums[warp_id as usize];
            let final_exclusive = g_scan[tid as usize] + warp_prefix;
            let out_idx = block_idx * NUM_BUCKETS as u32 + digit;
            offsets[out_idx as usize] = base_offset + reduction + final_exclusive;
        }
    }
}
