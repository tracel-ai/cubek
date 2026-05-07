use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum InterpolateError {
    #[error("Unsupported interpolation mode: {0}")]
    UnsupportedMode(String),

    #[error(
        "Interpolate expects 4D tensors (NHWC), but got input rank {input} and output rank {output}"
    )]
    InvalidRank { input: usize, output: usize },

    #[error("Batch size mismatch: input has {input} but output has {output}")]
    BatchMismatch { input: usize, output: usize },

    #[error("Channel count mismatch: input has {input} but output has {output}")]
    ChannelMismatch { input: usize, output: usize },

    #[error(
        "Shape mismatch: input shape {input:?} and output gradient shape {output:?} must match exactly"
    )]
    ShapeMismatch {
        input: Vec<usize>,
        output: Vec<usize>,
    },
}
