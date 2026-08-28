use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum InterpolateError {
    #[error("Unsupported interpolation mode: {0}")]
    UnsupportedMode(String),

    #[error(
        "Requested shared memory {requested} bytes exceeds the device limit of {available} bytes"
    )]
    SharedMemoryLimitExceeded { requested: usize, available: usize },

    #[error("Shared memory residence is not supported on CPU")]
    SharedMemoryUnsupportedOnCpu,

    #[error("Requested {requested} units per cube exceeds the device limit of {available} units")]
    UnitsPerCubeExceeded { requested: usize, available: usize },

    #[error("Interpolation config must use at least one plane per cube")]
    ZeroPlanesPerCube,

    #[error("Interpolation config must use at least one row per plane")]
    ZeroRowsPerPlane,

    #[error("Interpolation config must use at least one column per lane")]
    ZeroColsPerLane,

    #[error("Interpolation config channel block must contain at least one channel")]
    ZeroChannelBlock,

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

impl From<InterpolateError> for String {
    fn from(error: InterpolateError) -> Self {
        error.to_string()
    }
}
