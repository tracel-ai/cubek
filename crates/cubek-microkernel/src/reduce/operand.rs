use cubecl::tile::Dimension;

/// One of a reduction's operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOperand {
    /// `[BATCH, REDUCED]`: the rows reduced, along [`REDUCED`].
    Input,
    /// `[BATCH, SLOTS]`: the cells each row writes (one, or `k` for a top-k), the values or their
    /// indices.
    Out,
}

/// The rows: every axis of a reduction's input but the reduced one, as one dimension, each row
/// reduced on its own.
pub const BATCH: Dimension = Dimension(0);
/// The axis a row runs along, which the reduction takes away.
pub const REDUCED: Dimension = Dimension(1);
/// The cells a row writes, `k` for a top-k: the output's dimension in place of the reduced one.
pub const SLOTS: Dimension = Dimension(2);
/// A hypercube's partials, every part's carrier for a row side by side: the partials' dimension in
/// place of the reduced one.
pub const PARTIALS: Dimension = Dimension(3);
