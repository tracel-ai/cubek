use cubecl::tile::Dimension;

/// One of a reduction's operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOperand {
    /// `[ROWS, REDUCED]`: the rows reduced, along [`REDUCED`].
    Input,
    /// `[ROWS]`, or `[ROWS, k]` for a top-k: one cell per row, the values or their indices.
    Out,
}

/// The rows: each one reduced on its own.
pub const ROWS: Dimension = Dimension(0);
/// The reduced dimension.
pub const REDUCED: Dimension = Dimension(1);
