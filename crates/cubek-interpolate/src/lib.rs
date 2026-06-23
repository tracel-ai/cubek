pub mod components;
pub mod definition;
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
pub mod launch;
pub mod routines;

use crate::{
    definition::{InterpolateError, InterpolateMode, InterpolateOptions},
    launch::{InterpolateStrategy, interpolate_launch, interpolate_nearest_backward_launch},
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
    strategy: InterpolateStrategy,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    validate_strategy(client, &strategy)?;
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;

    interpolate_launch(client, input, output, options, strategy, dtype)
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

/// Checks if the strategy is valid for the current client
fn validate_strategy<R: Runtime>(
    client: &ComputeClient<R>,
    strategy: &InterpolateStrategy,
) -> Result<(), InterpolateError> {
    // If the client is not running on a CPU, we don't need to validate the strategy
    if client.properties().hardware.num_cpu_cores.is_none() {
        return Ok(());
    }

    // If the client is running on a CPU, we need to validate the strategy
    match strategy {
        InterpolateStrategy::GlobalMemoryStrategy(_) => Ok(()),
        InterpolateStrategy::SharedMemoryStrategy(_) => Err(InterpolateError::UnsupportedMode(
            "interpolation shared memory strategy is not supported on CPU".to_string(),
        )),
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
