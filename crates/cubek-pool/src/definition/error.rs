use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PoolError {
    #[error("Unsupported pooling mode: {mode}")]
    UnsupportedMode { mode: String },

    #[error("Invalid tensor rank: input {input} output {output}")]
    InvalidRank { input: usize, output: usize },

    #[error("Batch size mismatch: input has {input} but output has {output}")]
    BatchMismatch { input: usize, output: usize },

    #[error("Channel count mismatch: input has {input} but output has {output}")]
    ChannelMismatch { input: usize, output: usize },

    #[error("{tensor} spatial dimensions must be non-zero, got {actual:?}")]
    InvalidSpatialSize {
        tensor: &'static str,
        actual: Vec<usize>,
    },

    #[error("Output spatial shape mismatch: expected {expected:?} but got {actual:?}")]
    OutputSizeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Input gradient shape mismatch: expected {expected:?} but got {actual:?}")]
    InputGradientShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
