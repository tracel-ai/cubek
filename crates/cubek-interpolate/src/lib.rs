use core::result::Result;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*, std::FastDivmod};

use crate::interpolate_options::{InterpolateMode, InterpolateOptions};

mod bicubic;
mod bilinear;
mod error;
pub mod interpolate_options;
mod lanczos3;
mod nearest;
pub use error::InterpolateError;

use crate::bicubic::interpolate_bicubic_launch;
use crate::bilinear::interpolate_bilinear_launch;
use crate::lanczos3::interpolate_lanczos3_launch;
use crate::nearest::interpolate_nearest_launch;

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

    let result = match options.mode {
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
    };

    result
}

fn shape_divmod<R: Runtime>(binding: &TensorBinding<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}
