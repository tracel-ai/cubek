pub mod definition;
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
mod kernel;
pub mod tune_key;

pub use kernel::{InterpolateConfig, Residence};

use crate::{
    definition::{InterpolateError, InterpolateMode, InterpolateOptions},
    kernel::{interpolate_launch, interpolate_nearest_backward_launch},
};
use core::result::Result;
use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

/// Interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes.
///
/// Expects input in NHWC layout.
pub fn interpolate<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    config: InterpolateConfig,
    dtype: ElemType,
) -> Result<(), InterpolateError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_shape(&input.shape)?;
    validate_shape(&output.shape)?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;
    config.validate()?;

    interpolate_launch(client, input, output, options, dtype, config)
}

/// Backward interpolate operation
///
/// Note: only nearest mode is supported
///
/// Expects input in NHWC layout.
pub fn interpolate_backward<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    input_grad: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
) -> Result<(), InterpolateError> {
    validate_rank(input.shape.len(), input_grad.shape.len())?;
    validate_rank(out_grad.shape.len(), input_grad.shape.len())?;
    validate_shape(&input.shape)?;
    validate_shape(&out_grad.shape)?;
    validate_shape(&input_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &input_grad.shape)?;
    validate_nhwc_consistency(&out_grad.shape, &input_grad.shape)?;

    if input.shape != input_grad.shape {
        return Err(InterpolateError::ShapeMismatch {
            input: input.shape.to_vec(),
            input_grad: input_grad.shape.to_vec(),
        });
    }

    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => {
            interpolate_nearest_backward_launch(client, out_grad, input_grad, nearest_mode, dtype)
        }
        _ => Err(InterpolateError::UnsupportedMode(format!(
            "{:?} interpolation backward is not supported by JIT backend",
            options.mode
        ))),
    }
}

/// Check that both tensors are 4D (Batch, Height, Width, Channels).
fn validate_rank(input_rank: usize, output_rank: usize) -> Result<(), InterpolateError> {
    if input_rank != 4 || output_rank != 4 {
        return Err(InterpolateError::InvalidRank {
            input: input_rank,
            output: output_rank,
        });
    }
    Ok(())
}

fn validate_shape(shape: &[usize]) -> Result<(), InterpolateError> {
    for (axis, &size) in shape.iter().enumerate() {
        if size == 0 {
            return Err(InterpolateError::ZeroDimension {
                shape: shape.to_vec(),
                axis,
            });
        }
        if matches!(axis, 1 | 2) && size > i32::MAX as usize {
            return Err(InterpolateError::SpatialDimensionTooLarge {
                shape: shape.to_vec(),
                axis,
                size,
                max: i32::MAX as usize,
            });
        }
    }
    Ok(())
}

/// Check that Batch (0) and Channel (3) dimensions match.
/// Height (1) and Width (2) are allowed to differ for resizing.
fn validate_nhwc_consistency(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), InterpolateError> {
    if input_shape[0] != output_shape[0] {
        return Err(InterpolateError::BatchMismatch {
            input: input_shape[0],
            output: output_shape[0],
        });
    }

    if input_shape[3] != output_shape[3] {
        return Err(InterpolateError::ChannelMismatch {
            input: input_shape[3],
            output: output_shape[3],
        });
    }

    Ok(())
}
