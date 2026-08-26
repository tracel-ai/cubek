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
    /// The accumulator's innermost (column) extent in scalars.
    pub cols: usize,
    /// Contracted values one step consumes ([`Space::served`]).
    pub served: usize,
    /// How many sink cells one block column's vector lanes spread across.
    pub spread: usize,
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
        let cols = space.extent_at(rank - 1);
        let spread = if served > 1 { 1 } else { vw / aw };
        // A spread block column holds `spread` scalar sink cells in its lanes and rounds up;
        // otherwise a cell is `vw`-wide (or 1 at a folded step) and keeps counting whole lines.
        let nr = if spread > 1 {
            cols.div_ceil(spread)
        } else {
            cols / (if served > 1 { 1 } else { vw })
        };

        Self {
            kc: reduce_extents.iter().product(),
            mr: space.extent_at(rank - 2),
            nr,
            cols,
            space,
            reduce,
            reduce_extents,
            served,
            spread,
            lw,
            vw,
            aw,
        }
    }

    /// Whether a 2-D reading describes both operands, and the axes it reads them over.
    ///
    /// The question the leaf routes on. A contraction the operand carries as one run of axes has a
    /// `k` edge — one axis, several partitioning one, or a convolution's taps beside its channels
    /// — and reads as a matrix. One it does not is read a cell at a time.
    pub fn matrix_axes(&self, lhs: &Space, rhs: &Space) -> Option<(MatrixAxes, MatrixAxes)> {
        let lhs_axes = MatrixAxes::find(lhs, self.mr, self.kc)?;
        let rhs_axes = match self.served > 1 {
            true => MatrixAxes::find(rhs, self.cols, self.kc)?,
            false => MatrixAxes::find(rhs, self.kc, self.cols)?,
        };
        Some((lhs_axes, rhs_axes))
    }

    /// Which of the lhs's axes form the `mr x kc` matrix the 2-D nest reads it as.
    ///
    /// Asked, not stored: a gathered operand has no matrix at all, which is the whole reason the
    /// N-D nest exists, and this same shape is what it runs from. An operand whose contracted axis
    /// is partitioned reads as one `k` edge over several axes, and only the edges say which.
    pub fn lhs_axes(&self, lhs: &Space) -> MatrixAxes {
        MatrixAxes::of(lhs, self.mr, self.kc)
    }

    /// The rhs's twin. A folded step lines it along the contraction, so its matrix is `(col, k)`;
    /// at one served value it lines along the accumulator and reads `(k, col)`.
    pub fn rhs_axes(&self, rhs: &Space) -> MatrixAxes {
        match self.served > 1 {
            true => MatrixAxes::of(rhs, self.cols, self.kc),
            false => MatrixAxes::of(rhs, self.kc, self.cols),
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
    /// lines of `served * aw` (exactly one of the two exceeds 1), or `spread` sink cells. Past
    /// the budget a schedule rolls its loops rather than keeping the block in registers.
    pub fn scalars(&self) -> usize {
        self.mr * self.nr * self.served * self.aw * self.spread
    }

    /// Whether the lane fan-out's fixed extracts stay in step with the coordinate
    /// `lane_component` decodes on the flat walk.
    pub fn lane_index_exact(&self) -> bool {
        self.reduce.len() == 1
            || self.reduce_extents[self.reduce_extents.len() - 1].is_multiple_of(self.lw)
    }
}
