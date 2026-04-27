use crate::components::{ConvolutionOperation, global::args::RuntimeArgs};
use cubecl::{
    prelude::*,
    std::tensor::{into_contiguous_pitched, is_contiguous_pitched},
};
use cubek_matmul::{
    definition::{AvailableVectorSizes, Blueprint},
    launch::MatmulArgs,
    routines::Routine as MatmulRoutine,
};
use std::fmt::Display;

/// Specifications for a convolution routine.
///
/// A `Routine` is the convolution-side counterpart of `cubek_matmul::routines::Routine`:
/// it pairs a per-operation matmul routine with the metadata needed to wire the
/// kernel up (input args, optional layout fixups, vector-size filtering).
///
/// `Blueprint` and `Strategy` are surfaced as direct associated types so callers
/// don't have to reach through `MatmulRoutine` to bound them.
pub trait Routine {
    type Blueprint: Blueprint;
    type Strategy: Default + Display + Clone;

    type MatmulRoutine: MatmulRoutine<RuntimeArgs, Blueprint = Self::Blueprint, Strategy = Self::Strategy>;
    type Args: MatmulArgs<Config = RuntimeArgs>;

    /// Whether to select specialized load flow in tests. Should replace with something cleaner
    /// eventually, but this is nice and simple.
    const IS_SPECIALIZED: bool = false;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError>;

    fn filter_vector_sizes(vector_sizes: AvailableVectorSizes) -> AvailableVectorSizes {
        vector_sizes
    }
}

pub(crate) fn contiguous_pitched_layout<R: Runtime>(
    client: &ComputeClient<R>,
    binding: TensorBinding<R>,
    dtype: StorageType,
) -> Result<TensorBinding<R>, LaunchError> {
    let binding = if has_valid_layout(&binding) {
        binding
    } else {
        into_contiguous_pitched(client, binding, dtype).binding()
    };
    Ok(binding)
}

fn has_valid_layout<R: Runtime>(binding: &TensorBinding<R>) -> bool {
    let rank = binding.shape.len();
    let dim_c = rank - 1;
    binding.strides[dim_c] == 1
}

const TMA_STRIDE_ALIGN: usize = 16;

pub(crate) fn into_tensor_handle_tma<R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    dtype: StorageType,
    operation: ConvolutionOperation,
) -> Result<TensorBinding<R>, LaunchError> {
    let binding = if has_valid_layout_tma(&handle, dtype, operation) {
        handle
    } else {
        into_contiguous_pitched(client, handle, dtype).binding()
    };
    Ok(binding)
}

pub(crate) fn has_valid_layout_tma<R: Runtime>(
    binding: &TensorBinding<R>,
    dtype: StorageType,
    operation: ConvolutionOperation,
) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / dtype.size();
    let rank = binding.shape.len();
    let dim_c = rank - 1;

    let aligned = binding.strides[..dim_c]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = binding.strides[dim_c] == 1;

    let is_valid_wgrad = if operation == ConvolutionOperation::BackwardWeight {
        is_contiguous_pitched(&binding.shape, &binding.strides)
    } else {
        true
    };

    valid_layout && aligned && is_valid_wgrad
}
