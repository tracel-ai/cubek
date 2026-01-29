//! Launch routines for radix sort.
//!
//! Orchestrates the multi-pass LSD (Least Significant Digit) radix sort:
//! 1. **Transform**: Convert keys to sortable unsigned representation
//! 2. **Histogram**: Count digit occurrences per block
//! 3. **Scan**: Compute global write offsets via prefix sum
//! 4. **Scatter**: Redistribute elements to sorted positions
//! 5. **Transform back**: Convert keys back to original representation
//!
//! Repeats histogram/scan/scatter for each 8-bit digit (4 passes for 32-bit keys).

use crate::components::config::{KeyTransform, NUM_BUCKETS, SortStrategy};
use crate::components::key::SortKey;
use crate::error::SortError;
use crate::kernels::{histogram, scan, scatter, transform};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Sort keys in ascending order.
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

    sort_keys_u32::<R>(
        client,
        keys_in,
        keys_out,
        num_items,
        K::NUM_PASSES,
        K::TRANSFORM,
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

    sort_pairs_u32::<R>(
        client,
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_items,
        K::NUM_PASSES,
        K::TRANSFORM,
        strategy,
    )
}

/// Sort u32 keys using ping-pong buffering.
fn sort_keys_u32<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    num_passes: u32,
    key_transform: KeyTransform,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    let num_blocks = strategy.num_blocks(num_items);
    let elem_size = core::mem::size_of::<u32>();

    // Allocate temporary buffers
    let temp_keys = client.empty(num_items * elem_size);
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * elem_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    // Transform keys to sortable representation if needed
    let transformed_input = match key_transform {
        KeyTransform::None => None,
        KeyTransform::SignedInt | KeyTransform::Float => {
            let transformed = client.empty(num_items * elem_size);
            launch_transform_to_radix::<R>(
                client,
                keys_in.handle,
                &transformed,
                num_items,
                key_transform,
            )?;
            Some(transformed)
        }
    };

    // Determine the actual input for sorting
    let sort_input = transformed_input.as_ref().unwrap_or(keys_in.handle);

    // Ping-pong pattern:
    // Pass 0: sort_input → temp_keys
    // Pass 1: temp_keys → keys_out
    // Pass 2: keys_out → temp_keys
    // Pass 3: temp_keys → keys_out
    for pass in 0..num_passes {
        let (src, dst) = match (pass == 0, pass % 2 == 0) {
            (true, _) => (sort_input, &temp_keys),
            (false, true) => (keys_out.handle, &temp_keys),
            (false, false) => (&temp_keys, keys_out.handle),
        };

        launch_histogram::<R>(
            client,
            src,
            &histograms,
            num_items,
            num_blocks,
            pass,
            &strategy,
        )?;
        launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;
        launch_scatter_keys::<R>(
            client, src, dst, &offsets, num_items, num_blocks, pass, &strategy,
        )?;
    }

    // Transform keys back from sortable representation if needed
    if key_transform != KeyTransform::None {
        // Result is in keys_out, transform in-place
        launch_transform_from_radix::<R>(
            client,
            keys_out.handle,
            keys_out.handle,
            num_items,
            key_transform,
        )?;
    }

    Ok(())
}

/// Sort u32 key-value pairs using ping-pong buffering.
#[allow(clippy::too_many_arguments)]
fn sort_pairs_u32<R: Runtime>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    num_passes: u32,
    key_transform: KeyTransform,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    let num_blocks = strategy.num_blocks(num_items);
    let elem_size = core::mem::size_of::<u32>();

    let temp_keys = client.empty(num_items * elem_size);
    let temp_values = client.empty(num_items * elem_size);
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * elem_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    // Transform keys to sortable representation if needed
    let transformed_input = match key_transform {
        KeyTransform::None => None,
        KeyTransform::SignedInt | KeyTransform::Float => {
            let transformed = client.empty(num_items * elem_size);
            launch_transform_to_radix::<R>(
                client,
                keys_in.handle,
                &transformed,
                num_items,
                key_transform,
            )?;
            Some(transformed)
        }
    };

    // Determine the actual input for sorting
    let sort_input = transformed_input.as_ref().unwrap_or(keys_in.handle);

    for pass in 0..num_passes {
        let (k_src, k_dst, v_src, v_dst) = match (pass == 0, pass % 2 == 0) {
            (true, _) => (sort_input, &temp_keys, values_in.handle, &temp_values),
            (false, true) => (keys_out.handle, &temp_keys, values_out.handle, &temp_values),
            (false, false) => (&temp_keys, keys_out.handle, &temp_values, values_out.handle),
        };

        launch_histogram::<R>(
            client,
            k_src,
            &histograms,
            num_items,
            num_blocks,
            pass,
            &strategy,
        )?;
        launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;
        launch_scatter_pairs::<R>(
            client, k_src, k_dst, v_src, v_dst, &offsets, num_items, num_blocks, pass, &strategy,
        )?;
    }

    // Transform keys back from sortable representation if needed
    if key_transform != KeyTransform::None {
        launch_transform_from_radix::<R>(
            client,
            keys_out.handle,
            keys_out.handle,
            num_items,
            key_transform,
        )?;
    }

    Ok(())
}

// ============================================================================
// Kernel Launchers
// ============================================================================

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

    unsafe {
        histogram::histogram_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
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
    let shape = [num_blocks as usize * NUM_BUCKETS];
    let strides = [1];

    let hist_tensor = unsafe { TensorArg::from_raw_parts::<u32>(histograms, &shape, &strides, 1) };
    let offsets_tensor = unsafe { TensorArg::from_raw_parts::<u32>(offsets, &shape, &strides, 1) };

    unsafe {
        scan::scan_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(NUM_BUCKETS as u32),
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

    unsafe {
        scatter::scatter_keys_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
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

    unsafe {
        scatter::scatter_pairs_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
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

// ============================================================================
// Transform Kernel Launchers
// ============================================================================

const TRANSFORM_THREADS: u32 = 256;

fn launch_transform_to_radix<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    num_items: usize,
    key_transform: KeyTransform,
) -> Result<(), SortError> {
    let shape = [num_items];
    let strides = [1];

    let input_tensor = unsafe { TensorArg::from_raw_parts::<u32>(input, &shape, &strides, 1) };
    let output_tensor = unsafe { TensorArg::from_raw_parts::<u32>(output, &shape, &strides, 1) };

    #[allow(clippy::manual_div_ceil)]
    let num_blocks = (num_items as u32 + TRANSFORM_THREADS - 1) / TRANSFORM_THREADS;

    match key_transform {
        KeyTransform::None => Ok(()),
        KeyTransform::SignedInt => unsafe {
            transform::transform_i32_to_radix::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(num_blocks),
                CubeDim::new_1d(TRANSFORM_THREADS),
                input_tensor,
                output_tensor,
                ScalarArg::new(num_items as u32),
            )
            .map_err(SortError::Launch)
        },
        KeyTransform::Float => unsafe {
            transform::transform_f32_to_radix::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(num_blocks),
                CubeDim::new_1d(TRANSFORM_THREADS),
                input_tensor,
                output_tensor,
                ScalarArg::new(num_items as u32),
            )
            .map_err(SortError::Launch)
        },
    }
}

fn launch_transform_from_radix<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    num_items: usize,
    key_transform: KeyTransform,
) -> Result<(), SortError> {
    let shape = [num_items];
    let strides = [1];

    let input_tensor = unsafe { TensorArg::from_raw_parts::<u32>(input, &shape, &strides, 1) };
    let output_tensor = unsafe { TensorArg::from_raw_parts::<u32>(output, &shape, &strides, 1) };

    #[allow(clippy::manual_div_ceil)]
    let num_blocks = (num_items as u32 + TRANSFORM_THREADS - 1) / TRANSFORM_THREADS;

    match key_transform {
        KeyTransform::None => Ok(()),
        KeyTransform::SignedInt => unsafe {
            transform::transform_radix_to_i32::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(num_blocks),
                CubeDim::new_1d(TRANSFORM_THREADS),
                input_tensor,
                output_tensor,
                ScalarArg::new(num_items as u32),
            )
            .map_err(SortError::Launch)
        },
        KeyTransform::Float => unsafe {
            transform::transform_radix_to_f32::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(num_blocks),
                CubeDim::new_1d(TRANSFORM_THREADS),
                input_tensor,
                output_tensor,
                ScalarArg::new(num_items as u32),
            )
            .map_err(SortError::Launch)
        },
    }
}
