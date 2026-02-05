use cubek_matmul::launch::MatmulArgs;
use cubek_matmul::{definition::AvailableLineSizes, routines::Routine};

use cubecl::std::tensor::{TensorHandle, into_contiguous_pitched_ref, is_contiguous_pitched};

use cubecl::prelude::*;

use crate::components::{ConvolutionOperation, global::args::RuntimeArgs};

pub mod simple;
pub mod specialized;

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type Routine: Routine<RuntimeArgs>;
    type Args: MatmulArgs<Config = RuntimeArgs>;

    /// Whether to select specialized load flow in tests. Should replace with something cleaner
    /// eventually, but this is nice and simple.
    const IS_SPECIALIZED: bool = false;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError>;

    fn filter_line_sizes(line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        line_sizes
    }
}

pub(crate) fn into_tensor_handle<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let handle = if has_valid_layout(handle) {
        TensorHandle::from_ref(handle, dtype)
    } else {
        into_contiguous_pitched_ref(client, handle, dtype)?
    };
    Ok(handle)
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    handle.strides[dim_c] == 1
}

const TMA_STRIDE_ALIGN: usize = 16;

pub(crate) fn into_tensor_handle_tma<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorHandleRef<'_, R>,
    dtype: StorageType,
    operation: ConvolutionOperation,
) -> Result<TensorHandle<R>, LaunchError> {
    let handle = if has_valid_layout_tma(handle, operation) {
        TensorHandle::from_ref(handle, dtype)
    } else {
        into_contiguous_pitched_ref(client, handle, dtype)?
    };
    Ok(handle)
}

pub(crate) fn has_valid_layout_tma<R: Runtime>(
    handle: &TensorHandleRef<'_, R>,
    operation: ConvolutionOperation,
) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;
    let rank = handle.shape.len();
    let dim_c = rank - 1;

    let aligned = handle.strides[..dim_c]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = handle.strides[dim_c] == 1;

    let is_valid_wgrad = if operation == ConvolutionOperation::BackwardWeight {
        is_contiguous_pitched(handle.shape, handle.strides)
    } else {
        true
    };

    valid_layout && aligned && is_valid_wgrad
}
