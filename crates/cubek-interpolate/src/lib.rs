pub mod components;
pub mod definition;
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
pub mod launch;
pub mod routines;

use crate::{
    definition::{InterpolateError, InterpolateMode, InterpolateOptions, NearestMode},
    launch::{InterpolateStrategy, interpolate_launch, interpolate_nearest_backward_launch},
};
use core::result::Result;
use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};
use cubek_resample::{
    definition::{Kernel, Placement, Resample, Semiring},
    resample,
};

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
    strategy: InterpolateStrategy,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;

    //interpolate_launch(client, input, output, options, strategy, dtype)

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 1.0,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(1)
        .with_axis(2);

    resample(client, input, output, config, dtype);

    Ok(())
}

pub fn get_scale_and_offset(
    input_size: usize,
    output_size: usize,
    options: InterpolateOptions,
) -> (f32, f32) {
    let standard_scale = input_size as f32 / output_size as f32;

    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => match nearest_mode {
            NearestMode::Exact => (standard_scale, standard_scale / 2.0),
            NearestMode::Floor => (standard_scale, 0.0),
        },
        _ => {
            if options.align_corners {
                let scale_num = if output_size > 1 {
                    input_size.saturating_sub(1) as f32
                } else {
                    0.0
                };
                let scale_den = output_size.saturating_sub(1).max(1) as f32;

                (scale_num / scale_den, 0.0)
            } else {
                let offset = (standard_scale - 1.0) / 2.0;
                (standard_scale, offset)
            }
        }
    }
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
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_rank(out_grad.shape.len(), output.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;
    validate_nhwc_consistency(&out_grad.shape, &output.shape)?;

    if input.shape != output.shape {
        return Err(InterpolateError::ShapeMismatch {
            input: input.shape.to_vec(),
            output: output.shape.to_vec(),
        });
    }

    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => {
            interpolate_nearest_backward_launch(client, out_grad, output, nearest_mode, dtype)
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
