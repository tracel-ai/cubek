use core::result::Result;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

use crate::definition::{InterpolateMode, InterpolateOptions};

pub mod definition;
mod error;
mod modes;
pub use error::InterpolateError;

use crate::modes::bicubic::interpolate_bicubic_launch;
use crate::modes::bilinear::interpolate_bilinear_launch;
use crate::modes::lanczos3::interpolate_lanczos3_launch;
use crate::modes::nearest::interpolate_nearest_launch;

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

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
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let _align_corners = options.align_corners;

    match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_launch(client, input, output, dtype),
        InterpolateMode::Bilinear => {
            interpolate_bilinear_launch(client, input, output, _align_corners, dtype)
        }
        InterpolateMode::Bicubic => {
            interpolate_bicubic_launch(client, input, output, _align_corners, dtype)
        }
        InterpolateMode::Lanczos3 => {
            interpolate_lanczos3_launch(client, input, output, _align_corners, dtype)
        }
    }
}
