use crate::components::config::{NUM_BUCKETS, SortStrategy};
use crate::components::key::{SortKey, num_passes};
use crate::error::SortError;
use crate::kernels::{histogram, scan, scatter};
use cubecl::prelude::*;
use cubecl::server::Handle;

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

    sort_impl::<R, K>(
        client,
        keys_in.handle,
        keys_out.handle,
        None,
        None,
        num_items,
        strategy,
    )
}

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

    sort_impl::<R, K>(
        client,
        keys_in.handle,
        keys_out.handle,
        Some(values_in.handle),
        Some(values_out.handle),
        num_items,
        strategy,
    )
}

fn sort_impl<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: &Handle,
    keys_out: &Handle,
    values_in: Option<&Handle>,
    values_out: Option<&Handle>,
    num_items: usize,
    strategy: SortStrategy,
) -> Result<(), SortError> {
    let num_blocks = strategy.num_blocks(num_items);
    let elem_size = core::mem::size_of::<u32>();
    let has_values = values_in.is_some();
    let num_passes = num_passes::<K>();

    let temp_keys = client.empty(num_items * elem_size);
    let temp_values = if has_values {
        Some(client.empty(num_items * elem_size))
    } else {
        None
    };
    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * elem_size;
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    let last_pass = num_passes - 1;

    for pass in 0..num_passes {
        let is_first = pass == 0;
        let is_last = pass == last_pass;

        let (k_src, k_dst) = match (is_first, pass % 2 == 0) {
            (true, _) => (keys_in, &temp_keys),
            (false, true) => (keys_out, &temp_keys),
            (false, false) => (&temp_keys, keys_out),
        };

        let (v_src, v_dst) = if has_values {
            match (is_first, pass % 2 == 0) {
                (true, _) => (values_in.unwrap(), temp_values.as_ref().unwrap()),
                (false, true) => (values_out.unwrap(), temp_values.as_ref().unwrap()),
                (false, false) => (temp_values.as_ref().unwrap(), values_out.unwrap()),
            }
        } else {
            (keys_in, keys_out)
        };

        // First pass: read K, transform to u32
        // Middle passes: read u32, write u32 (identity)
        // Last pass: read u32, transform to K
        if is_first {
            launch_histogram::<R, K>(
                client,
                k_src,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        } else {
            launch_histogram::<R, u32>(
                client,
                k_src,
                &histograms,
                num_items,
                num_blocks,
                pass,
                &strategy,
            )?;
        }

        launch_scan::<R>(client, &histograms, &offsets, num_blocks)?;

        if is_first && is_last {
            launch_scatter::<R, K, K>(
                client, k_src, k_dst, v_src, v_dst, &offsets, num_items, num_blocks, pass,
                &strategy, has_values,
            )?;
        } else if is_first {
            launch_scatter::<R, K, u32>(
                client, k_src, k_dst, v_src, v_dst, &offsets, num_items, num_blocks, pass,
                &strategy, has_values,
            )?;
        } else if is_last {
            launch_scatter::<R, u32, K>(
                client, k_src, k_dst, v_src, v_dst, &offsets, num_items, num_blocks, pass,
                &strategy, has_values,
            )?;
        } else {
            launch_scatter::<R, u32, u32>(
                client, k_src, k_dst, v_src, v_dst, &offsets, num_items, num_blocks, pass,
                &strategy, has_values,
            )?;
        }
    }

    Ok(())
}

fn launch_histogram<R: Runtime, K: SortKey>(
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
        unsafe { TensorArg::from_raw_parts::<K>(keys, &keys_shape, &keys_strides, 1) };
    let hist_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1) };

    unsafe {
        histogram::histogram_kernel::launch_unchecked::<K, R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(strategy.threads_per_block),
            keys_tensor,
            hist_tensor,
            ScalarArg::new(num_items as u32),
            ScalarArg::new(pass),
            strategy.items_per_thread,
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
fn launch_scatter<R: Runtime, KIn: SortKey, KOut: SortKey>(
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
    has_values: bool,
) -> Result<(), SortError> {
    let items_shape = [num_items];
    let items_strides = [1];
    let offsets_shape = [num_blocks as usize * NUM_BUCKETS];
    let offsets_strides = [1];

    let keys_in_tensor =
        unsafe { TensorArg::from_raw_parts::<KIn>(keys_in, &items_shape, &items_strides, 1) };
    let keys_out_tensor =
        unsafe { TensorArg::from_raw_parts::<KOut>(keys_out, &items_shape, &items_strides, 1) };
    let values_in_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(values_in, &items_shape, &items_strides, 1) };
    let values_out_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(values_out, &items_shape, &items_strides, 1) };
    let offsets_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(offsets, &offsets_shape, &offsets_strides, 1) };

    unsafe {
        scatter::scatter_kernel::launch_unchecked::<KIn, KOut, R>(
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
            has_values,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}
