//! Launch routines for radix sort.
//!
//! This module orchestrates the multi-pass radix sort algorithm.

use crate::components::config::{NUM_BUCKETS, SortStrategy};
use crate::components::key::SortKey;
use crate::error::SortError;
use crate::kernels::{histogram, scan, scatter};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Sort keys in ascending order.
///
/// Currently only supports u32 keys directly. Other types will be supported
/// via transformation kernels.
pub fn sort_keys<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    if num_items == 0 {
        return Ok(());
    }

    // For now, we only support u32 directly
    // TODO: Add transformation kernels for i32, f32
    sort_keys_u32::<R>(
        client,
        keys_in,
        keys_out,
        num_items,
        K::NUM_PASSES,
        strategy,
    )
}

/// Sort key-value pairs by key in ascending order.
pub fn sort_pairs<R: Runtime, K: SortKey, V: Numeric>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    if num_items == 0 {
        return Ok(());
    }

    // For now, we only support u32 keys and values directly
    // TODO: Add transformation kernels for other types
    sort_pairs_u32::<R>(
        client,
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_items,
        K::NUM_PASSES,
        strategy,
    )
}

/// Internal implementation for u32 keys.
fn sort_keys_u32<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    num_passes: u32,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    let num_blocks = strategy.num_blocks(num_items);

    // Allocate temporary buffers
    let elem_size = core::mem::size_of::<u32>();
    let temp_keys = client.empty(num_items * elem_size);
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * elem_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    // For ping-pong we need references that work properly
    // Pass 0: keys_in -> temp_keys
    // Pass 1: temp_keys -> keys_out
    // Pass 2: keys_out -> temp_keys
    // Pass 3: temp_keys -> keys_out
    // So for 4 passes (even count), result is in keys_out

    for pass in 0..num_passes {
        let use_temp_as_output = pass % 2 == 0;
        let is_first_pass = pass == 0;

        // Launch histogram kernel
        if is_first_pass {
            launch_histogram::<R>(
                client,
                keys_in.handle,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else if use_temp_as_output {
            // Input is keys_out
            launch_histogram::<R>(
                client,
                keys_out.handle,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            // Input is temp_keys
            launch_histogram::<R>(
                client,
                &temp_keys,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }

        // Launch scan kernel
        launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;

        // Launch scatter kernel
        if is_first_pass {
            // Input: keys_in, Output: temp_keys
            launch_scatter_keys::<R>(
                client,
                keys_in.handle,
                &temp_keys,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else if use_temp_as_output {
            // Input: keys_out, Output: temp_keys
            launch_scatter_keys::<R>(
                client,
                keys_out.handle,
                &temp_keys,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            // Input: temp_keys, Output: keys_out
            launch_scatter_keys::<R>(
                client,
                &temp_keys,
                keys_out.handle,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }
    }

    Ok(())
}

/// Internal implementation for u32 key-value pairs.
#[allow(clippy::too_many_arguments)]
fn sort_pairs_u32<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    num_passes: u32,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    let num_blocks = strategy.num_blocks(num_items);

    let elem_size = core::mem::size_of::<u32>();
    let temp_keys = client.empty(num_items * elem_size);
    let temp_values = client.empty(num_items * elem_size);
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * elem_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    for pass in 0..num_passes {
        let use_temp_as_output = pass % 2 == 0;
        let is_first_pass = pass == 0;

        // Histogram
        if is_first_pass {
            launch_histogram::<R>(
                client,
                keys_in.handle,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else if use_temp_as_output {
            launch_histogram::<R>(
                client,
                keys_out.handle,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            launch_histogram::<R>(
                client,
                &temp_keys,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }

        launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;

        // Scatter
        if is_first_pass {
            launch_scatter_pairs::<R>(
                client,
                keys_in.handle,
                &temp_keys,
                values_in.handle,
                &temp_values,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else if use_temp_as_output {
            launch_scatter_pairs::<R>(
                client,
                keys_out.handle,
                &temp_keys,
                values_out.handle,
                &temp_values,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            launch_scatter_pairs::<R>(
                client,
                &temp_keys,
                keys_out.handle,
                &temp_values,
                values_out.handle,
                &offsets,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }
    }

    Ok(())
}

fn launch_histogram<R: Runtime>(
    client: &ComputeClient<R>,
    keys: &Handle,
    histograms: &Handle,
    num_items: usize,
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
) -> Result<(), SortError> {
    let keys_shape = [num_items];
    let keys_strides = [1];
    let hist_shape = [num_blocks as usize * NUM_BUCKETS];
    let hist_strides = [1];

    let keys_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(keys, &keys_shape, &keys_strides, 1) };
    let hist_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1) };

    let cube_dim = CubeDim::new_1d(strategy.threads_per_block);
    let cube_count = CubeCount::new_1d(num_blocks);

    unsafe {
        histogram::histogram_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            keys_tensor,
            hist_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            strategy.items_per_thread,
            NUM_BUCKETS as u32,
            strategy.threads_per_block,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

fn launch_scan<R: Runtime>(
    client: &ComputeClient<R>,
    histograms: &Handle,
    offsets: &Handle,
    num_blocks: u32,
) -> Result<(), SortError> {
    let total_size = num_blocks as usize * NUM_BUCKETS;
    let shape = [total_size];
    let strides = [1];

    let hist_tensor = unsafe { TensorArg::from_raw_parts::<u32>(histograms, &shape, &strides, 1) };
    let offsets_tensor = unsafe { TensorArg::from_raw_parts::<u32>(offsets, &shape, &strides, 1) };

    // Launch with NUM_BUCKETS (256) threads for parallel scan
    let cube_dim = CubeDim::new_1d(NUM_BUCKETS as u32);
    let cube_count = CubeCount::new_1d(1);

    unsafe {
        scan::scan_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            hist_tensor,
            offsets_tensor,
            ScalarArg::new(num_blocks),
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_keys<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: &Handle,
    keys_out: &Handle,
    offsets: &Handle,
    num_items: usize,
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
) -> Result<(), SortError> {
    let keys_shape = [num_items];
    let keys_strides = [1];
    let offsets_shape = [num_blocks as usize * NUM_BUCKETS];
    let offsets_strides = [1];

    let keys_in_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(keys_in, &keys_shape, &keys_strides, 1) };
    let keys_out_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(keys_out, &keys_shape, &keys_strides, 1) };
    let offsets_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(offsets, &offsets_shape, &offsets_strides, 1) };

    let cube_dim = CubeDim::new_1d(strategy.threads_per_block);
    let cube_count = CubeCount::new_1d(num_blocks);

    unsafe {
        scatter::scatter_keys_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            keys_in_tensor,
            keys_out_tensor,
            offsets_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            strategy.items_per_thread,
            NUM_BUCKETS as u32,
            strategy.threads_per_block,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_pairs<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: &Handle,
    keys_out: &Handle,
    values_in: &Handle,
    values_out: &Handle,
    offsets: &Handle,
    num_items: usize,
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
) -> Result<(), SortError> {
    let items_shape = [num_items];
    let items_strides = [1];
    let offsets_shape = [num_blocks as usize * NUM_BUCKETS];
    let offsets_strides = [1];

    let keys_in_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(keys_in, &items_shape, &items_strides, 1) };
    let keys_out_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(keys_out, &items_shape, &items_strides, 1) };
    let values_in_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(values_in, &items_shape, &items_strides, 1) };
    let values_out_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(values_out, &items_shape, &items_strides, 1) };
    let offsets_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(offsets, &offsets_shape, &offsets_strides, 1) };

    let cube_dim = CubeDim::new_1d(strategy.threads_per_block);
    let cube_count = CubeCount::new_1d(num_blocks);

    unsafe {
        scatter::scatter_pairs_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            keys_in_tensor,
            keys_out_tensor,
            values_in_tensor,
            values_out_tensor,
            offsets_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            strategy.items_per_thread,
            NUM_BUCKETS as u32,
            strategy.threads_per_block,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}
