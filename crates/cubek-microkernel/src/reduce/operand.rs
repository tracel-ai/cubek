use cubecl::tile::Axis;

/// One of a reduction's operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOperand {
    /// `[o, r]`: the rows reduced, along `r`.
    Input,
    /// `[o]`, or `[o, k]` for a top-k: one cell per row, the values or their indices.
    Out,
}

/// The rows: each one reduced on its own.
pub const O: Axis = Axis(0);
/// The reduced axis.
pub const R: Axis = Axis(1);
