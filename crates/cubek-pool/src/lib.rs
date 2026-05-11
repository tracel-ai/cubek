use core::result::Result;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

pub mod definition;
mod kernel;

use crate::definition::{PoolError, PoolMode};
use crate::kernel::{
    backward::{
        adaptive_avg_pool2d_backward_launch, avg_pool2d_backward_launch,
        max_pool2d_with_indices_backward_launch,
    },
    forward::{
        adaptive_avg_pool2d_launch, avg_pool2d_launch, max_pool2d_launch,
        max_pool2d_with_indices_launch,
    },
};

/// Pool2d public wrapper
///
/// Expects input in NHWC layout.
pub fn pool2d<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;

    match mode {
        PoolMode::Max(max_options) => max_pool2d_launch(client, input, output, max_options, dtype),
        PoolMode::Avg(avg_options) => avg_pool2d_launch(client, input, output, avg_options, dtype),
        PoolMode::AdaptiveAvg(adaptive_avg_options) => {
            adaptive_avg_pool2d_launch(client, input, output, adaptive_avg_options, dtype)
        }
    }
}

/// Pool2d with indices public wrapper
///
/// Expects input in NHWC layout. Output indices are expected to be in the same layout as well.
pub fn pool2d_with_indices<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    indices: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_rank(input.shape.len(), indices.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;
    validate_nhwc_consistency(&input.shape, &indices.shape)?;

    match mode {
        PoolMode::Max(max_options) => {
            max_pool2d_with_indices_launch(client, input, output, indices, max_options, dtype)
        }
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

/// Pool2d backward public wrapper
///
/// Expects input and output gradients in NHWC layout.
pub fn pool2d_backward<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len())?;
    validate_rank(input.shape.len(), in_grad.shape.len())?;
    validate_nhwc_consistency(&input.shape, &out_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &in_grad.shape)?;

    match mode {
        PoolMode::Avg(avg_options) => {
            avg_pool2d_backward_launch(client, input, out_grad, in_grad, avg_options, dtype)
        }
        PoolMode::AdaptiveAvg(adaptive_avg_options) => adaptive_avg_pool2d_backward_launch(
            client,
            input,
            out_grad,
            in_grad,
            adaptive_avg_options,
            dtype,
        ),
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

/// Pool2d backward with indices public wrapper
///
/// Expects input and output gradients in NHWC layout. Output indices are expected to be in the same layout as well.
#[allow(clippy::too_many_arguments)]
pub fn pool2d_with_indices_backward<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    indices: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
    indices_dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len())?;
    validate_rank(input.shape.len(), in_grad.shape.len())?;
    validate_rank(input.shape.len(), indices.shape.len())?;
    validate_nhwc_consistency(&input.shape, &out_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &in_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &indices.shape)?;

    match mode {
        PoolMode::Max(max_options) => max_pool2d_with_indices_backward_launch(
            client,
            input,
            out_grad,
            indices,
            in_grad,
            max_options,
            dtype,
            indices_dtype,
        ),
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

/// Check that both tensors are 4D (Batch, Height, Width, Channels).
fn validate_rank(input_rank: usize, output_rank: usize) -> Result<(), PoolError> {
    if input_rank != 4 || output_rank != 4 {
        return Err(PoolError::InvalidRank {
            input: input_rank,
            output: output_rank,
        });
    }
    Ok(())
}

/// Check that Batch (0) and Channel (3) dimensions match.
/// Height (1) and Width (2) are allowed to differ for resizing.
fn validate_nhwc_consistency(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), PoolError> {
    if input_shape[0] != output_shape[0] {
        return Err(PoolError::BatchMismatch {
            input: input_shape[0],
            output: output_shape[0],
        });
    }

    if input_shape[3] != output_shape[3] {
        return Err(PoolError::ChannelMismatch {
            input: input_shape[3],
            output: output_shape[3],
        });
    }

    Ok(())
}
