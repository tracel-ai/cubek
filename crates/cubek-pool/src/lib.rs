use core::result::Result;

use cubecl::{client::Client, prelude::TensorBinding, prelude::*};

#[cfg(feature = "benchmarks")]
pub mod eval;

pub mod definition;
mod kernel;

use crate::definition::{PoolError, PoolMode};
use crate::kernel::{
    backward::{
        adaptive_avg_pool2d_backward_launch, adaptive_avg_pool3d_backward_launch,
        avg_pool2d_backward_launch, max_pool2d_with_indices_backward_launch,
    },
    forward::{
        adaptive_avg_pool2d_launch, adaptive_avg_pool3d_launch, avg_pool2d_launch,
        max_pool2d_launch, max_pool2d_with_indices_launch,
    },
};

/// Pool2d public wrapper
///
/// Expects input in NHWC layout.
pub fn pool2d(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    mode: PoolMode<2>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len(), 4)?;
    validate_batch_channel_consistency(&input.shape, &output.shape)?;

    match mode {
        PoolMode::Max(max_options) => max_pool2d_launch(client, input, output, max_options, dtype),
        PoolMode::Avg(avg_options) => avg_pool2d_launch(client, input, output, avg_options, dtype),
        PoolMode::AdaptiveAvg(adaptive_avg_options) => {
            adaptive_avg_pool2d_launch(client, input, output, adaptive_avg_options, dtype)
        }
    }
}

/// Pool3d public wrapper.
///
/// Expects matching floating-point input and output tensors in NDHWC layout. Only
/// [`PoolMode::AdaptiveAvg`] is supported, and its configured output size must match the output
/// tensor's spatial shape.
///
/// # Errors
///
/// Returns [`PoolError::InvalidRank`] unless both tensors have rank five,
/// [`PoolError::BatchMismatch`] or [`PoolError::ChannelMismatch`] when their outer dimensions do
/// not agree, [`PoolError::InvalidSpatialSize`] when either tensor has a zero spatial dimension,
/// [`PoolError::OutputSizeMismatch`] when the configured spatial size differs from the output
/// tensor, and [`PoolError::UnsupportedMode`] for any other pooling mode.
pub fn pool3d(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    mode: PoolMode<3>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len(), 5)?;
    validate_batch_channel_consistency(&input.shape, &output.shape)?;

    match mode {
        PoolMode::AdaptiveAvg(options) => {
            validate_spatial_size(&input.shape, "input")?;
            validate_spatial_size(&output.shape, "output")?;
            validate_output_size(&output.shape, &options.output_size)?;
            adaptive_avg_pool3d_launch(client, input, output, dtype)
        }
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

/// Pool2d with indices public wrapper
///
/// Expects input in NHWC layout. Output indices are expected to be in the same layout as well.
pub fn pool2d_with_indices(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    indices: TensorBinding,
    mode: PoolMode<2>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len(), 4)?;
    validate_rank(input.shape.len(), indices.shape.len(), 4)?;
    validate_batch_channel_consistency(&input.shape, &output.shape)?;
    validate_batch_channel_consistency(&input.shape, &indices.shape)?;

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
pub fn pool2d_backward(
    client: &Client,
    input: TensorBinding,
    out_grad: TensorBinding,
    in_grad: TensorBinding,
    mode: PoolMode<2>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len(), 4)?;
    validate_rank(input.shape.len(), in_grad.shape.len(), 4)?;
    validate_batch_channel_consistency(&input.shape, &out_grad.shape)?;
    validate_batch_channel_consistency(&input.shape, &in_grad.shape)?;

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

/// Pool3d backward public wrapper.
///
/// Expects matching floating-point input, output-gradient, and input-gradient tensors in NDHWC
/// layout. Only [`PoolMode::AdaptiveAvg`] is supported. The configured output size must match the
/// output gradient, and the input gradient must have exactly the input shape.
///
/// # Errors
///
/// Returns [`PoolError::InvalidRank`] unless every tensor has rank five,
/// [`PoolError::BatchMismatch`] or [`PoolError::ChannelMismatch`] when their outer dimensions do
/// not agree, [`PoolError::InputGradientShapeMismatch`] when the input-gradient shape differs from
/// the input, [`PoolError::InvalidSpatialSize`] when the input or output gradient has a zero spatial
/// dimension, [`PoolError::OutputSizeMismatch`] when the configured spatial size differs from the
/// output gradient, and [`PoolError::UnsupportedMode`] for any other pooling mode.
pub fn pool3d_backward(
    client: &Client,
    input: TensorBinding,
    out_grad: TensorBinding,
    in_grad: TensorBinding,
    mode: PoolMode<3>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len(), 5)?;
    validate_rank(input.shape.len(), in_grad.shape.len(), 5)?;
    validate_batch_channel_consistency(&input.shape, &out_grad.shape)?;
    validate_batch_channel_consistency(&input.shape, &in_grad.shape)?;

    if input.shape != in_grad.shape {
        return Err(PoolError::InputGradientShapeMismatch {
            expected: input.shape.to_vec(),
            actual: in_grad.shape.to_vec(),
        });
    }

    match mode {
        PoolMode::AdaptiveAvg(options) => {
            validate_spatial_size(&input.shape, "input")?;
            validate_spatial_size(&out_grad.shape, "output gradient")?;
            validate_output_size(&out_grad.shape, &options.output_size)?;
            adaptive_avg_pool3d_backward_launch(client, out_grad, in_grad, dtype)
        }
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

/// Pool2d backward with indices public wrapper
///
/// Expects input and output gradients in NHWC layout. Output indices are expected to be in the same layout as well.
#[allow(clippy::too_many_arguments)]
pub fn pool2d_with_indices_backward(
    client: &Client,
    input: TensorBinding,
    out_grad: TensorBinding,
    indices: TensorBinding,
    in_grad: TensorBinding,
    mode: PoolMode<2>,
    dtype: ElemType,
    indices_dtype: ElemType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len(), 4)?;
    validate_rank(input.shape.len(), in_grad.shape.len(), 4)?;
    validate_rank(input.shape.len(), indices.shape.len(), 4)?;
    validate_batch_channel_consistency(&input.shape, &out_grad.shape)?;
    validate_batch_channel_consistency(&input.shape, &in_grad.shape)?;
    validate_batch_channel_consistency(&input.shape, &indices.shape)?;

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

fn validate_rank(
    input_rank: usize,
    output_rank: usize,
    expected_rank: usize,
) -> Result<(), PoolError> {
    if input_rank != expected_rank || output_rank != expected_rank {
        return Err(PoolError::InvalidRank {
            input: input_rank,
            output: output_rank,
        });
    }
    Ok(())
}

/// Check that the batch and final channel dimensions match.
fn validate_batch_channel_consistency(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), PoolError> {
    if input_shape[0] != output_shape[0] {
        return Err(PoolError::BatchMismatch {
            input: input_shape[0],
            output: output_shape[0],
        });
    }

    let input_channel = input_shape[input_shape.len() - 1];
    let output_channel = output_shape[output_shape.len() - 1];
    if input_channel != output_channel {
        return Err(PoolError::ChannelMismatch {
            input: input_channel,
            output: output_channel,
        });
    }

    Ok(())
}

fn validate_output_size(output_shape: &[usize], expected: &[usize; 3]) -> Result<(), PoolError> {
    let actual = &output_shape[1..4];
    if actual != expected {
        return Err(PoolError::OutputSizeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

fn validate_spatial_size(shape: &[usize], tensor: &'static str) -> Result<(), PoolError> {
    let actual = &shape[1..4];
    if actual.contains(&0) {
        return Err(PoolError::InvalidSpatialSize {
            tensor,
            actual: actual.to_vec(),
        });
    }
    Ok(())
}
