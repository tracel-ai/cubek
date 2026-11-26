use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum InterpolateError {
    #[error(transparent)]
    Declined(#[from] cubecl::tile::launch::Decline),

    #[error("Unsupported interpolation mode: {0}")]
    UnsupportedMode(String),

    #[error("Tensor shape {shape:?} has a zero-sized dimension at axis {axis}")]
    ZeroDimension { shape: Vec<usize>, axis: usize },

    #[error(
        "Tensor shape {shape:?} has spatial dimension {axis} of size {size}, which exceeds the maximum of {max}"
    )]
    SpatialDimensionTooLarge {
        shape: Vec<usize>,
        axis: usize,
        size: usize,
        max: usize,
    },

    #[error(
        "Interpolate expects 4D tensors (NHWC), but got input rank {input} and output rank {output}"
    )]
    InvalidRank { input: usize, output: usize },

    #[error("Batch size mismatch: input has {input} but output has {output}")]
    BatchMismatch { input: usize, output: usize },

    #[error("Channel count mismatch: input has {input} but output has {output}")]
    ChannelMismatch { input: usize, output: usize },

    #[error(
        "Shape mismatch: input shape {input:?} and input gradient shape {input_grad:?} must match exactly"
    )]
    ShapeMismatch {
        input: Vec<usize>,
        input_grad: Vec<usize>,
    },
}

/// The decline a launch returns stays a refusal, which a test skips where the launch is
/// unsupported and fails on where it is broken; any other error is a kernel that did not run.
#[cfg(feature = "cpu-reference")]
impl From<InterpolateError> for cubek_test_utils::KernelResultError {
    fn from(error: InterpolateError) -> Self {
        match error {
            InterpolateError::Declined(decline) => Self::Refused(decline),
            other => Self::NotRun(other.to_string()),
        }
    }
}

impl From<InterpolateError> for String {
    fn from(error: InterpolateError) -> Self {
        error.to_string()
    }
}
