//! The shape a contraction is cut to: how the accumulator block is divided, how deep the
//! contraction runs, and the line widths the operands and the accumulator sit at.
//!
//! Derived once at the leaf's entry and handed to whichever schedule runs it, so the 2-D nest,
//! the gathered nest and the separable one cannot disagree about the block they are filling or
//! the budget they are measuring it against.

use crate::*;

#[derive(Clone, Debug)]
pub(super) struct ContractShape {
    /// The accumulator's space: the batch axes, then the row, then the column.
    pub space: Space,
    /// The axes the operands contract against the accumulator.
    pub reduce: Vec<Axis>,
    /// Their extents, taken off the operands' merged space rather than the accumulator's: a
    /// contracted axis is by definition absent from the accumulator, and an axis only one operand
    /// spans still has to be walked.
    pub reduce_extents: Vec<usize>,
    /// The contraction depth, every contracted axis multiplied out.
    pub kc: usize,
    /// The block's rows.
    pub mr: usize,
    /// Its columns, counted in cells.
    pub nr: usize,
    /// Contracted values one step consumes ([`Space::served`]).
    pub served: usize,
    /// The lhs's line width.
    pub lw: usize,
    /// The rhs's, and so the block's.
    pub vw: usize,
    /// The accumulator's.
    pub aw: usize,
}

impl ContractShape {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lhs: &Space,
        rhs: &Space,
        space: Space,
        served: usize,
        lw: usize,
        vw: usize,
        aw: usize,
    ) -> Self {
        let rank = space.rank();
        let merged = Space::merge(&[lhs, rhs]);
        let reduce = Space::contracted(&[lhs, rhs], &space).to_vec();
        let reduce_extents = reduce
            .iter()
            .map(|&axis| merged.extent(axis))
            .collect::<Vec<_>>();
        // A folded step holds partials of one cell in the block's lanes, so its cells are one
        // column wide; otherwise a cell is the rhs's line.
        let cell_width = if served > 1 { 1 } else { vw };

        Self {
            kc: reduce_extents.iter().product(),
            mr: space.extent_at(rank - 2),
            nr: space.extent_at(rank - 1) / cell_width,
            space,
            reduce,
            reduce_extents,
            served,
            lw,
            vw,
            aw,
        }
    }

    /// The accumulator's batch axes: everything above the row and the column.
    pub fn batch_extents(&self) -> Vec<usize> {
        (0..self.space.rank() - 2)
            .map(|p| self.space.extent_at(p))
            .collect()
    }

    /// How many batch matrices a nest walks.
    pub fn matrices(&self) -> usize {
        self.batch_extents().iter().product()
    }

    /// The block's size in scalars, which is what [`RegisterBlock::budget`] counts: `mr * nr`
    /// lines of `served * aw` (exactly one of the two exceeds 1). Past the budget a schedule
    /// rolls its loops rather than keeping the block in registers.
    pub fn scalars(&self) -> usize {
        self.mr * self.nr * self.served * self.aw
    }
}
