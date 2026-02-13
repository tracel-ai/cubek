use crate::error::SortError;
use crate::key::{Radix, SortKey};
use crate::routines::histogram::HistogramBlueprint;
use crate::routines::scan::ScanBlueprint;
use crate::routines::scatter::ValuesMode;
use crate::routines::{NUM_BUCKETS, histogram, scan, scatter};
use crate::{SortOrder, SortOutput, SortValues};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_std::tensor::layout::linear::{LinearLayout, LinearLayoutArgs, LinearViewLaunch};

#[derive(Clone, Debug)]
pub struct SortStrategy {
    pub items_per_thread: u32,
    pub threads_per_block: u32,
}

impl Default for SortStrategy {
    fn default() -> Self {
        // 512 threads × 16 items = 8192 items per block
        // Larger blocks reduce scan overhead at the cost of occupancy
        Self {
            items_per_thread: 12,
            threads_per_block: 512,
        }
    }
}

impl SortStrategy {
    pub fn items_per_block(&self) -> u32 {
        self.items_per_thread * self.threads_per_block
    }

    pub fn num_blocks(&self, num_items: usize) -> u32 {
        num_items.div_ceil(self.items_per_block() as usize) as u32
    }
}

/// Convert an element size to a StorageType using unsigned integers.
/// We use unsigned ints as the "carrier" type since we just need to copy bytes.
fn value_dtype_from_size(elem_size: usize) -> StorageType {
    match elem_size {
        1 => u8::as_type_native().unwrap(),
        2 => u16::as_type_native().unwrap(),
        4 => u32::as_type_native().unwrap(),
        8 => u64::as_type_native().unwrap(),
        _ => panic!("Unsupported value element size: {elem_size} bytes (supported: 1, 2, 4, 8)"),
    }
}

pub fn sort<'a, R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<'a, R>,
    values: SortValues<'a, R>,
    order: SortOrder,
) -> Result<SortOutput, SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let strategy = SortStrategy::default();

    let num_keys = keys_in.shape[0];
    let key_size = core::mem::size_of::<K>();

    // Get value element size from the tensor, or use u32 size for indices/none
    let value_size = match &values {
        SortValues::None => core::mem::size_of::<u32>(),
        SortValues::Tensor(t) => t.elem_size,
        SortValues::Indices => core::mem::size_of::<u32>(),
    };

    // Allocate output buffers
    let keys_out = client.empty(keys_in.shape[0] * key_size);
    let values_out = match &values {
        SortValues::None => None,
        SortValues::Tensor(_) | SortValues::Indices => Some(client.empty(num_keys * value_size)),
    };

    sort_impl::<R, K>(
        client,
        &keys_in,
        &keys_out,
        &values,
        values_out.as_ref(),
        value_size,
        strategy,
        order,
    )?;

    Ok(SortOutput {
        keys: keys_out,
        values: values_out,
    })
}

#[allow(clippy::too_many_arguments)]
fn sort_impl<'a, R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: &TensorHandleRef<R>,
    keys_out: &Handle,
    values: &SortValues<'a, R>,
    values_out: Option<&Handle>,
    value_size: usize,
    strategy: SortStrategy,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let num_keys = keys_in.shape[0];
    let num_blocks = strategy.num_blocks(num_keys);
    let radix_size = core::mem::size_of::<K::Radix>();

    // One pass per radix byte.
    let num_passes = core::mem::size_of::<K>() as u32;

    let plane_dim = client.properties().hardware.plane_size_min;
    let num_planes = strategy.threads_per_block.div_ceil(plane_dim);

    // Determine values mode for first pass vs subsequent passes
    let (first_pass_mode, later_pass_mode) = match values {
        SortValues::None => (ValuesMode::None, ValuesMode::None),
        SortValues::Tensor(_) => (ValuesMode::Tensor, ValuesMode::Tensor),
        SortValues::Indices => (ValuesMode::Indices, ValuesMode::Tensor),
    };

    // Temp buffer for radix keys
    let temp_keys = client.empty(num_keys * radix_size);
    let temp_values = if first_pass_mode != ValuesMode::None {
        Some(client.empty(num_keys * value_size))
    } else {
        None
    };

    let histogram_size = (num_blocks as usize) * NUM_BUCKETS * core::mem::size_of::<u32>();
    let histograms = client.empty(histogram_size);
    let offsets = client.empty(histogram_size);

    let last_pass = num_passes - 1;

    let cont_shape = [num_keys];
    let cont_stride = [1];

    for pass in 0..num_passes {
        let is_first = pass == 0;
        let is_last = pass == last_pass;

        // Buffer selection for keys
        let (k_src_handle, k_src_shape, k_src_strides, k_dst) = if is_first && is_last {
            (keys_in.handle, keys_in.shape, keys_in.strides, keys_out)
        } else if is_first {
            (keys_in.handle, keys_in.shape, keys_in.strides, &temp_keys)
        } else {
            match pass % 2 == 0 {
                true => (keys_out, &cont_shape[..], &cont_stride[..], &temp_keys),
                false => (&temp_keys, &cont_shape[..], &cont_stride[..], keys_out),
            }
        };

        // Buffer selection for values
        let (v_src_handle, v_src_shape, v_src_strides, v_dst) =
            if first_pass_mode != ValuesMode::None {
                let v_out = values_out.unwrap();
                if is_first && is_last {
                    // For Indices mode, v_src is unused but we need a valid handle
                    let v_in = match values {
                        SortValues::Tensor(t) => (t.handle, t.shape, t.strides),
                        _ => (keys_in.handle, keys_in.shape, keys_in.strides), // Dummy, won't be read
                    };
                    (v_in.0, v_in.1, v_in.2, v_out)
                } else if is_first {
                    let v_in = match values {
                        SortValues::Tensor(t) => (t.handle, t.shape, t.strides),
                        _ => (keys_in.handle, keys_in.shape, keys_in.strides),
                    };
                    (v_in.0, v_in.1, v_in.2, temp_values.as_ref().unwrap())
                } else {
                    match pass % 2 == 0 {
                        true => (
                            v_out,
                            &cont_shape[..],
                            &cont_stride[..],
                            temp_values.as_ref().unwrap(),
                        ),
                        false => (
                            temp_values.as_ref().unwrap(),
                            &cont_shape[..],
                            &cont_stride[..],
                            v_out,
                        ),
                    }
                }
            } else {
                // Dummy values when not sorting pairs
                (keys_in.handle, keys_in.shape, keys_in.strides, keys_out)
            };

        let histogram_blueprint = HistogramBlueprint {
            threads_per_block: strategy.threads_per_block,
            items_per_thread: strategy.items_per_thread,
        };

        // Histogram phase
        if is_first {
            launch_histogram::<R, K, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                &histograms,
                num_blocks,
                pass,
                histogram_blueprint,
            )?;
        } else {
            launch_histogram::<R, K::Radix, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                &histograms,
                num_blocks,
                pass,
                histogram_blueprint,
            )?;
        }

        let scan_blueprint = ScanBlueprint { scan_dim: 256 };
        launch_scan::<R>(client, &histograms, &offsets, num_blocks, scan_blueprint)?;

        let reverse_output = order.is_descending() && is_last;
        let values_mode = if is_first {
            first_pass_mode
        } else {
            later_pass_mode
        };

        if is_first && is_last {
            launch_scatter::<R, K, K, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                k_dst,
                v_src_handle,
                v_src_shape,
                v_src_strides,
                v_dst,
                &offsets,
                num_blocks,
                pass,
                &strategy,
                values_mode,
                num_planes,
                reverse_output,
                value_size,
            )?;
        } else if is_first {
            launch_scatter::<R, K, K::Radix, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                k_dst,
                v_src_handle,
                v_src_shape,
                v_src_strides,
                v_dst,
                &offsets,
                num_blocks,
                pass,
                &strategy,
                values_mode,
                num_planes,
                reverse_output,
                value_size,
            )?;
        } else if is_last {
            launch_scatter::<R, K::Radix, K, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                k_dst,
                v_src_handle,
                v_src_shape,
                v_src_strides,
                v_dst,
                &offsets,
                num_blocks,
                pass,
                &strategy,
                values_mode,
                num_planes,
                reverse_output,
                value_size,
            )?;
        } else {
            launch_scatter::<R, K::Radix, K::Radix, K::Radix>(
                client,
                k_src_handle,
                k_src_shape,
                k_src_strides,
                k_dst,
                v_src_handle,
                v_src_shape,
                v_src_strides,
                v_dst,
                &offsets,
                num_blocks,
                pass,
                &strategy,
                values_mode,
                num_planes,
                reverse_output,
                value_size,
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
    num_blocks: u32,
    pass: u32,
    blueprint: HistogramBlueprint,
) -> Result<(), SortError> {
    let hist_shape = [num_blocks as usize * NUM_BUCKETS];
    let hist_strides = [1];

    let keys_view = linear_view::<K, R>(client, keys, keys_shape, keys_strides);
    let hist_tensor =
        unsafe { TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1) };

    unsafe {
        histogram::histogram_kernel::launch_unchecked::<K, Rx, R>(
            client,
            CubeCount::new_1d(num_blocks),
            CubeDim::new_1d(blueprint.threads_per_block),
            keys_view,
            hist_tensor,
            ScalarArg::new(pass),
            blueprint,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

fn linear_view<'a, E: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    handle: &'a Handle,
    shape: &[usize],
    strides: &[usize],
) -> LinearViewLaunch<'a, R> {
    let layout = LinearLayoutArgs::from_shape_strides(client, shape, strides, 1);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(handle, shape[0], 1, core::mem::size_of::<E>())
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

fn launch_scan<R: Runtime>(
    client: &ComputeClient<R>,
    histograms: &Handle,
    offsets: &Handle,
    num_blocks: u32,
    blueprint: ScanBlueprint,
) -> Result<(), SortError> {
    let elem_size = core::mem::size_of::<u32>();
    let hist_shape = [num_blocks as usize * NUM_BUCKETS];
    let hist_strides = [1];
    let digit_shape = [NUM_BUCKETS];
    let digit_strides = [1];

    let digit_totals = client.empty(NUM_BUCKETS * elem_size);
    let digit_prefixes = client.empty(NUM_BUCKETS * elem_size);

    unsafe {
        let hist_tensor =
            TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1);
        let totals_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_totals, &digit_shape, &digit_strides, 1);

        scan::scan_sum_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(NUM_BUCKETS as u32),
            CubeDim::new_1d(blueprint.scan_dim),
            hist_tensor,
            totals_tensor,
            ScalarArg::new(num_blocks),
            blueprint,
        )
        .map_err(SortError::Launch)?;
    }

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
            blueprint,
        )
        .map_err(SortError::Launch)?;
    }

    unsafe {
        let hist_tensor =
            TensorArg::from_raw_parts::<u32>(histograms, &hist_shape, &hist_strides, 1);
        let prefixes_tensor =
            TensorArg::from_raw_parts::<u32>(&digit_prefixes, &digit_shape, &digit_strides, 1);
        let offsets_tensor =
            TensorArg::from_raw_parts::<u32>(offsets, &hist_shape, &hist_strides, 1);

        scan::scan_offsets::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(NUM_BUCKETS as u32),
            CubeDim::new_1d(blueprint.scan_dim),
            hist_tensor,
            prefixes_tensor,
            offsets_tensor,
            ScalarArg::new(num_blocks),
            blueprint,
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
    num_blocks: u32,
    pass: u32,
    strategy: &SortStrategy,
    values_mode: ValuesMode,
    num_planes: u32,
    reverse_output: bool,
    value_size: usize,
) -> Result<(), SortError> {
    let offsets_shape = [num_blocks as usize * NUM_BUCKETS];
    let offsets_strides = [1];

    // Convert size to StorageType for kernel dispatch
    let value_dtype = value_dtype_from_size(value_size);

    let keys_in_view = linear_view::<KIn, R>(client, keys_in, keys_in_shape, keys_in_strides);
    let values_in_view = linear_view_dynamic::<R>(
        client,
        values_in,
        values_in_shape,
        values_in_strides,
        value_size,
    );

    let keys_out_tensor =
        unsafe { TensorArg::from_raw_parts::<KOut>(keys_out, keys_in_shape, keys_in_strides, 1) };
    let values_out_tensor = unsafe {
        TensorArg::from_raw_parts_and_size(
            values_out,
            keys_in_shape,
            keys_in_strides,
            1,
            value_size,
        )
    };
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
            ScalarArg::new(pass),
            ScalarArg::new(reverse_output as u32),
            strategy.items_per_thread,
            values_mode,
            num_planes,
            strategy.items_per_block(),
            value_dtype,
        )
        .map_err(SortError::Launch)?;
    }
    Ok(())
}

fn linear_view_dynamic<'a, R: Runtime>(
    client: &ComputeClient<R>,
    handle: &'a Handle,
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = LinearLayoutArgs::from_shape_strides(client, shape, strides, 1);
    let buffer = unsafe { ArrayArg::from_raw_parts_and_size(handle, shape[0], 1, elem_size) };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}
