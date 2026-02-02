use crate::components::config::{NUM_BUCKETS, SortStrategy};
use crate::components::key::{Radix, SortKey};
use crate::error::SortError;
use crate::kernels::{histogram, scan, scatter};
use crate::SortOrder;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_std::tensor::layout::linear::{LinearLayout, LinearLayoutArgs, LinearViewLaunch};

pub fn sort_keys<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: SortStrategy,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    if num_items == 0 {
        return Ok(());
    }

    sort_impl::<R, K>(
        client,
        &keys_in,
        keys_out.handle,
        None,
        None,
        num_items,
        strategy,
        order,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sort_pairs<R: Runtime, K: SortKey, V: Numeric>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: SortStrategy,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    if num_items == 0 {
        return Ok(());
    }

    sort_impl::<R, K>(
        client,
        &keys_in,
        keys_out.handle,
        Some(&values_in),
        Some(values_out.handle),
        num_items,
        strategy,
        order,
    )
}

#[allow(clippy::too_many_arguments)]
fn sort_impl<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: &TensorHandleRef<R>,
    keys_out: &Handle,
    values_in: Option<&TensorHandleRef<R>>,
    values_out: Option<&Handle>,
    num_items: usize,
    strategy: SortStrategy,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let num_blocks = strategy.num_blocks(num_items);
    // Use radix type size for intermediate buffers
    let radix_size = core::mem::size_of::<K::Radix>();
    let value_size = core::mem::size_of::<u32>();
    let has_values = values_in.is_some();
    // Number of passes = number of bytes in the key type (not radix type)
    // This is correct because we only need to sort the meaningful bytes.
    // For u8: 1 pass, u16: 2 passes, u32: 4 passes
    let num_passes = core::mem::size_of::<K>() as u32;
    let plane_dim = client.properties().hardware.plane_size_min;
    let num_planes = strategy.num_planes(plane_dim);

    // Temp buffer for radix keys (may be u32 or u64)
    let temp_keys = client.empty(num_items * radix_size);
    let temp_values = if has_values {
        Some(client.empty(num_items * value_size))
    } else {
        None
    };
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * value_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    let last_pass = num_passes - 1;

    // Contiguous shape/strides for temp buffers and outputs
    let contiguous_shape = [num_items];
    let contiguous_strides = [1];

    for pass in 0..num_passes {
        let is_first = pass == 0;
        let is_last = pass == last_pass;

        // Buffer selection for keys - first pass uses original input, others use contiguous temps
        let (k_src_handle, k_src_shape, k_src_strides, k_dst) = if is_first && is_last {
            // Single pass: read from input (strided), write directly to output
            (keys_in.handle, keys_in.shape, keys_in.strides, keys_out)
        } else if is_first {
            // First pass: read from input (strided), write to temp
            (keys_in.handle, keys_in.shape, keys_in.strides, &temp_keys)
        } else {
            // Subsequent passes: alternate between temp and output (all contiguous)
            match pass % 2 == 0 {
                true => (keys_out, &contiguous_shape[..], &contiguous_strides[..], &temp_keys),
                false => (&temp_keys, &contiguous_shape[..], &contiguous_strides[..], keys_out),
            }
        };

        let (v_src_handle, v_src_shape, v_src_strides, v_dst) = if has_values {
            if is_first && is_last {
                let v_in = values_in.unwrap();
                (v_in.handle, v_in.shape, v_in.strides, values_out.unwrap())
            } else if is_first {
                let v_in = values_in.unwrap();
                (v_in.handle, v_in.shape, v_in.strides, temp_values.as_ref().unwrap())
            } else {
                match pass % 2 == 0 {
                    true => (values_out.unwrap(), &contiguous_shape[..], &contiguous_strides[..], temp_values.as_ref().unwrap()),
                    false => (temp_values.as_ref().unwrap(), &contiguous_shape[..], &contiguous_strides[..], values_out.unwrap()),
                }
            }
        } else {
            // Dummy values when not sorting pairs - won't be used
            (keys_in.handle, keys_in.shape, keys_in.strides, keys_out)
        };

        if is_first {
            launch_histogram::<R, K, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            // K::Radix is SortKey with Radix = K::Radix (u32 or u64)
            launch_histogram::<R, K::Radix, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }

        // Choose scan strategy based on input size:
        // - Small inputs (< 256 blocks): single-block scan has lower overhead
        // - Large inputs (>= 256 blocks): cooperative scan has better parallelism
        if num_blocks < 256 {
            launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;
        } else {
            launch_scan_cooperative::<R>(client, &histograms, &offsets, num_blocks, 256)?;
        }

        // Only reverse output on the final pass when sorting descending
        let reverse_output = order.is_descending() && is_last;

        if is_first && is_last {
            // Single pass: K -> K
            launch_scatter::<R, K, K, K::Radix>(
                client, k_src_handle, k_src_shape, k_src_strides, k_dst,
                v_src_handle, v_src_shape, v_src_strides, v_dst,
                &offsets, num_items, num_blocks, pass,
                &strategy, has_values, num_planes, reverse_output,
            )?;
        } else if is_first {
            // First pass: K -> K::Radix
            launch_scatter::<R, K, K::Radix, K::Radix>(
                client, k_src_handle, k_src_shape, k_src_strides, k_dst,
                v_src_handle, v_src_shape, v_src_strides, v_dst,
                &offsets, num_items, num_blocks, pass,
                &strategy, has_values, num_planes, reverse_output,
            )?;
        } else if is_last {
            // Last pass: K::Radix -> K
            launch_scatter::<R, K::Radix, K, K::Radix>(
                client, k_src_handle, k_src_shape, k_src_strides, k_dst,
                v_src_handle, v_src_shape, v_src_strides, v_dst,
                &offsets, num_items, num_blocks, pass,
                &strategy, has_values, num_planes, reverse_output,
            )?;
        } else {
            // Middle pass: K::Radix -> K::Radix
            launch_scatter::<R, K::Radix, K::Radix, K::Radix>(
                client, k_src_handle, k_src_shape, k_src_strides, k_dst,
                v_src_handle, v_src_shape, v_src_strides, v_dst,
                &offsets, num_items, num_blocks, pass,
                &strategy, has_values, num_planes, reverse_output,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_histogram<R: Runtime, K: SortKey<Radix = Rx>, Rx: Radix>(
    client: &ComputeClient<R>,
    keys: &Handle,
    keys_shape: &[usize],
    keys_strides: &[usize],
    histograms: &Handle,
    num_items: usize,
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
) -> Result<(), SortError> {
    let hist_shape = [num_blocks as usize * NUM_BUCKETS];
    let hist_strides = [1];

    // Create LinearView for keys - handles strided access automatically
    let keys_view = linear_view::<K, R>(client, keys, keys_shape, keys_strides, num_items);
    let hist_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1) };

    unsafe {
        histogram::histogram_kernel::launch_unchecked::<K, Rx, R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
            keys_view,
            hist_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            strategy.items_per_thread,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

/// Helper to create a LinearView launch arg from a handle and shape/strides
fn linear_view<'a, E: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    handle: &'a Handle,
    shape: &[usize],
    strides: &[usize],
    num_items: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = LinearLayoutArgs::from_shape_strides(client, shape, strides, 1);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(handle, num_items, 1, core::mem::size_of::<E>())
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

/// Three-phase cooperative scan for better parallelism.
/// Phase A: Sum digit totals (256 blocks, SCAN_DIM threads each)
/// Phase B: Cross-digit prefix sum (1 block, 256 threads)
/// Phase C: Within-digit offsets (256 blocks, SCAN_DIM threads each)
fn launch_scan_cooperative<R: Runtime>(
    client: &ComputeClient<R>,
    histograms: &Handle,
    offsets: &Handle,
    num_blocks: u32,
    scan_dim: u32,
) -> Result<(), SortError> {
    let elem_size = core::mem::size_of::<u32>();
    let hist_shape = [num_blocks as usize * NUM_BUCKETS];
    let hist_strides = [1];
    let digit_shape = [NUM_BUCKETS];
    let digit_strides = [1];

    // Allocate temp buffers for digit totals and prefixes
    let digit_totals = client.empty(NUM_BUCKETS * elem_size);
    let digit_prefixes = client.empty(NUM_BUCKETS * elem_size);

    // Phase A: Compute digit totals
    unsafe {
        let hist_tensor =
            TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1);
        let totals_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_totals, &digit_shape, &digit_strides, 1);

        scan::scan_sum_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(NUM_BUCKETS as u32),
            CubeDim::new_1d(scan_dim),
            hist_tensor,
            totals_tensor,
            ScalarArg::new(num_blocks),
            scan_dim,
        )
        .map_err(SortError::Launch)?;
    }

    // Phase B: Cross-digit prefix sum
    unsafe {
        let totals_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_totals, &digit_shape, &digit_strides, 1);
        let prefixes_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_prefixes, &digit_shape, &digit_strides, 1);

        scan::scan_prefix_totals_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(NUM_BUCKETS as u32),
            totals_tensor,
            prefixes_tensor,
        )
        .map_err(SortError::Launch)?;
    }

    // Phase C: Within-digit cooperative offsets
    unsafe {
        let hist_tensor =
            TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1);
        let prefixes_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_prefixes, &digit_shape, &digit_strides, 1);
        let offsets_tensor =
            TensorArg::from_raw_parts::<u32>(offsets, &hist_shape, &hist_strides, 1);

        scan::scan_offsets_cooperative_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(NUM_BUCKETS as u32),
            CubeDim::new_1d(scan_dim),
            hist_tensor,
            prefixes_tensor,
            offsets_tensor,
            ScalarArg::new(num_blocks),
            scan_dim,
        )
        .map_err(SortError::Launch)?;
    }

    Ok(())
}

/// Single-block scan - fallback for small num_blocks.
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
fn launch_scatter<R: Runtime, KIn: SortKey<Radix = Rx>, KOut: SortKey<Radix = Rx>, Rx: Radix>(
    client: &ComputeClient<R>,
    keys_in: &Handle,
    keys_in_shape: &[usize],
    keys_in_strides: &[usize],
    keys_out: &Handle,
    values_in: &Handle,
    values_in_shape: &[usize],
    values_in_strides: &[usize],
    values_out: &Handle,
    offsets: &Handle,
    num_items: usize,
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
    has_values: bool,
    num_planes: u32,
    reverse_output: bool,
) -> Result<(), SortError> {
    let items_shape = [num_items];
    let items_strides = [1];
    let offsets_shape = [num_blocks as usize * NUM_BUCKETS];
    let offsets_strides = [1];

    // Create LinearViews for inputs - handles strided access automatically
    let keys_in_view = linear_view::<KIn, R>(client, keys_in, keys_in_shape, keys_in_strides, num_items);
    let values_in_view = linear_view::<u32, R>(client, values_in, values_in_shape, values_in_strides, num_items);

    // Outputs are always contiguous
    let keys_out_tensor =
        unsafe { TensorArg::from_raw_parts::<KOut>(keys_out, &items_shape, &items_strides, 1) };
    let values_out_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(values_out, &items_shape, &items_strides, 1) };
    let offsets_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(offsets, &offsets_shape, &offsets_strides, 1) };

    unsafe {
        scatter::scatter_kernel::launch_unchecked::<KIn, KOut, Rx, R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
            keys_in_view,
            keys_out_tensor,
            values_in_view,
            values_out_tensor,
            offsets_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            ScalarArg::new(reverse_output as u32),
            strategy.items_per_thread,
            has_values,
            num_planes,
            strategy.items_per_block(),
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}
