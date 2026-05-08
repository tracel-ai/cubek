use core::result::Result;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;
pub mod definition;
mod kernel;

use crate::definition::{PoolError, PoolMode};
use crate::kernel::forward::max_pool2d_launch;

/// MaxPool2d public wrapper
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
        _ => unimplemented!("Only MaxPool2d is implemented currently"),
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
