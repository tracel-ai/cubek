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
}
