//! What the register block reads: one line of an operand, at a position.
//!
//! A plain operand hands its line back. A scaled one multiplies its scales in first, and does it
//! here rather than under a view because a scale line covers several value lines: which lane of it
//! a line takes has to be a constant, and only a caller walking a comptime run knows that. `run`
//! is that position, and an operand with nothing to index by it ignores it.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// One operand's lines as the contraction reads them.
#[cube]
pub(crate) trait Lines<E: Numeric, V: Size>: CubeType {
    /// The line at `pos`, folded with whatever this operand carries beside its values. `run` names
    /// the position within the run one scale line covers, so the lane it selects is a constant.
    fn line(&self, pos: Coords2d, #[comptime] run: usize) -> Vector<E, V>;
}

#[cube]
impl<'a, E: Numeric, V: Size> Lines<E, V> for MaskedView<'a, Vector<E, V>, Coords2d> {
    fn line(&self, pos: Coords2d, #[comptime] _run: usize) -> Vector<E, V> {
        self.read(pos)
    }
}
