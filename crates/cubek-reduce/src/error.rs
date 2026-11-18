use thiserror::Error;

/// Why a reduction did not launch.
#[derive(Error, Debug, Clone)]
pub enum ReduceError {
    /// The kernel the strategy names declines the problem on this device, for this reason.
    #[error("A generic validation error: {details}")]
    Validation { details: &'static str },
    /// The axis is not a dim of the input.
    #[error("The provided axis ({axis}) must be smaller than the input tensor rank ({rank}).")]
    InvalidAxis { axis: usize, rank: usize },
    /// The reduced dim is shorter than the `k` a top-k keeps.
    #[error(
        "The input reduce axis length (currently {axis_length:?}) should be at least k ({k:?})."
    )]
    ReduceAxisTooSmall { axis_length: usize, k: usize },
    /// The output's shape is not the input's with the reduced dim one wide (`k` wide for a top-k).
    #[error("The output shape (currently {output_shape:?}) should be {expected_shape:?}.")]
    MismatchOutputShape {
        expected_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// The indices output's shape differs from the values output's.
    #[error(
        "The indices shape (currently {indices_shape:?}) should match the values shape ({values_shape:?})."
    )]
    MismatchIndicesShape {
        values_shape: Vec<usize>,
        indices_shape: Vec<usize>,
    },
    /// The indices output's strides differ from the values output's.
    #[error(
        "The indices strides (currently {indices_strides:?}) should match the values strides ({values_strides:?})."
    )]
    MismatchIndicesStrides {
        values_strides: Vec<usize>,
        indices_strides: Vec<usize>,
    },
    /// The operation has no index to write beside its values.
    #[error(
        "The operation {operation} has no index to report; reduce_with_indices only supports TopK, ArgTopK, Max, ArgMax, Min and ArgMin."
    )]
    IndicesUnsupported { operation: &'static str },
}
