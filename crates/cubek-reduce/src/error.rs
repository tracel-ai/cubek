use cubecl::tile::launch::Decline;
use thiserror::Error;

/// Why a reduction did not launch.
#[derive(Error, Debug, Clone)]
pub enum ReduceError {
    /// The kernel the strategy names cannot run the problem on this device, for the reason the
    /// decline names.
    #[error("The reduce kernel cannot run this problem on this device: {0}")]
    Declined(#[from] Decline),
    /// The axis is not a dim of the input.
    #[error("The provided axis ({axis}) must be smaller than the input tensor rank ({rank}).")]
    InvalidAxis { axis: usize, rank: usize },
    /// A top-k keeps no value: `k` is 0.
    #[error("A top-k keeps at least one value; k is 0.")]
    EmptyTopK,
    /// The reduced dim is shorter than the cells a row writes: `k` for a top-k, one for every other
    /// operation, which an empty dim is too short for.
    #[error(
        "The input reduce axis length (currently {axis_length:?}) should be at least the {k:?} a row writes."
    )]
    ReduceAxisTooSmall { axis_length: usize, k: usize },
    /// The output's shape is not the input's with the reduced dim one wide (`k` wide for a top-k).
    #[error("The output shape (currently {output_shape:?}) should be {expected_shape:?}.")]
    MismatchOutputShape {
        /// The shape the output should have, one extent an axis, shown in order (Arrival Usage).
        expected_shape: Vec<usize>,
        /// The output's shape, shown in order (Arrival Usage).
        output_shape: Vec<usize>,
    },
    /// The indices output's shape differs from the values output's.
    #[error(
        "The indices shape (currently {indices_shape:?}) should match the values shape ({values_shape:?})."
    )]
    MismatchIndicesShape {
        /// The values output's shape, one extent an axis, shown in order (Arrival Usage).
        values_shape: Vec<usize>,
        /// The indices output's shape, shown in order (Arrival Usage).
        indices_shape: Vec<usize>,
    },
    /// The operation has no index to write beside its values.
    #[error(
        "The operation {operation} has no index to report; Reduce::execute_with_indices only supports TopK, ArgTopK, Max, ArgMax, Min and ArgMin."
    )]
    IndicesUnsupported { operation: &'static str },
}
