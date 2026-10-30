use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PrefixScanError {
    #[error("Unsupported rank {rank}, expected 1")]
    UnsupportedRank { rank: usize },

    #[error("Input / output shape mismatch: input {input:?} output {output:?}")]
    InputOutputShapeMismatch {
        input: Vec<usize>,
        output: Vec<usize>,
    },

    #[error("Missing support for device_memory_scope")]
    MissingDeviceMemoryScope,

    #[error("Input size would lead to empty cubes")]
    CubeCountCanNotBeFactored,

    #[error("Input size would lead to a number of cubes which overflows u32::MAX")]
    CubeCountOverflowsU32,
}
