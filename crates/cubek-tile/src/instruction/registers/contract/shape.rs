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
    /// The accumulator's own matrix ([`MatrixAxes::accumulator`]), so the one place that decides
    /// its edges is the one place every reader asks.
    pub acc_axes: MatrixAxes,
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
    /// Contracted values one step consumes ([`Space::contracted_per_step`]).
    pub contracted_per_step: usize,
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
        contracted_per_step: usize,
        lw: usize,
        vw: usize,
        aw: usize,
    ) -> Self {
        let merged = Space::merge(&[lhs, rhs]);
        let acc_axes = MatrixAxes::accumulator(&space, lhs);
        let reduce = Space::contracted(&[lhs, rhs], &space).to_vec();
        let reduce_extents = reduce
            .iter()
            .map(|&axis| merged.extent(axis))
            .collect::<Vec<_>>();
        let cols = acc_axes.cols(&space);
        let spread = if contracted_per_step > 1 { 1 } else { vw / aw };
        // A spread block column rounds up, since its lanes hold whole sink cells and the last
        // one may be short; every other cell width divides the column edge exactly.
        let cell = column_cell_width(contracted_per_step, spread, vw);
        let nr = if spread > 1 {
            cols.div_ceil(cell)
        } else {
            cols / cell
        };

        let mr = acc_axes.rows(&space);

        Self {
            kc: reduce_extents.iter().product(),
            mr,
            nr,
            cols,
            space,
            acc_axes,
            reduce,
            reduce_extents,
            contracted_per_step,
            spread,
            lw,
            vw,
            aw,
        }
    }

    /// Whether a 2-D reading describes both operands, and the axes it reads them over.
    ///
    /// The question the leaf routes on. A contraction the operand carries as one run of axes has a
    /// `k` edge (one axis, several partitioning one, or a convolution's taps beside its channels)
    /// reads as a matrix. One it does not is read a cell at a time.
    pub(crate) fn matrix_axes(&self, lhs: &Space, rhs: &Space) -> Option<(MatrixAxes, MatrixAxes)> {
        let lhs_axes = MatrixAxes::find(lhs, self.mr, self.kc)?;
        let rhs_axes = match self.contracted_per_step > 1 {
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
    pub(crate) fn lhs_axes(&self, lhs: &Space) -> MatrixAxes {
        MatrixAxes::of(lhs, self.mr, self.kc)
    }

    /// The rhs's twin. A folded step lines it along the contraction, so its matrix is `(col, k)`;
    /// at one contracted value per step it lines along the accumulator and reads `(k, col)`.
    pub(crate) fn rhs_axes(&self, rhs: &Space) -> MatrixAxes {
        match self.contracted_per_step > 1 {
            true => MatrixAxes::of(rhs, self.cols, self.kc),
            false => MatrixAxes::of(rhs, self.kc, self.cols),
        }
    }

    /// The accumulator's column axes with their extents.
    pub(crate) fn column_edge(&self) -> Vec<(Axis, usize)> {
        (self.acc_axes.col_split..self.space.rank())
            .map(|p| (self.space.axis_at(p), self.space.extent_at(p)))
            .collect()
    }

    /// The contracted axes with theirs, which the accumulator cannot size: a contracted axis is by
    /// definition absent from it, so the extents come off the operands' merged space.
    pub(crate) fn reduce_edge(&self) -> Vec<(Axis, usize)> {
        self.reduce
            .iter()
            .copied()
            .zip(self.reduce_extents.iter().copied())
            .collect()
    }

    /// How many of the accumulator's innermost scalars one block column holds
    /// ([`column_cell_width`]).
    pub(crate) fn cell_width(&self) -> usize {
        column_cell_width(self.contracted_per_step, self.spread, self.vw)
    }

    /// The extents `row` unravels over: the axes between the batch prefix and the columns, in the
    /// accumulator's own order. One axis under [`MatrixAxes::accumulator`], which puts the row
    /// edge immediately above the column group, so the unravel is the identity there.
    pub(crate) fn row_extents(&self) -> Vec<usize> {
        line_extents(
            &self.space,
            1,
            self.acc_axes.row_split,
            self.acc_axes.col_split,
        )
    }

    /// The extents `col` unravels over: the column group, the innermost counted in the cells one
    /// block column holds rather than in scalars, so the product is `nr`.
    ///
    /// Several axes wherever the lhs stops the column group short of the row edge, which is every
    /// contraction whose accumulator carries axes no operand pairs it over — a depthwise
    /// convolution's `[batch, out_h, out_w, channel]` against a filter spanning the channel and
    /// the taps alone.
    pub(crate) fn column_line_extents(&self) -> Vec<usize> {
        line_extents(
            &self.space,
            self.cell_width(),
            self.acc_axes.col_split,
            self.space.rank(),
        )
    }

    /// The accumulator's batch axes: everything above the row edge.
    pub(crate) fn batch_extents(&self) -> Vec<usize> {
        (0..self.acc_axes.row_split)
            .map(|p| self.space.extent_at(p))
            .collect()
    }

    /// How many batch matrices a nest walks.
    pub fn matrices(&self) -> usize {
        self.batch_extents().iter().product()
    }

    /// The block's size in scalars, which is what [`RegisterBlock::budget`] counts: `mr * nr`
    /// lines of `contracted_per_step * aw` (exactly one of the two exceeds 1), or `spread` sink cells. Past
    /// the budget a schedule rolls its loops rather than keeping the block in registers.
    pub fn scalars(&self) -> usize {
        self.mr * self.nr * self.contracted_per_step * self.aw * self.spread
    }

    /// Whether the lane fan-out's fixed extracts stay in step with the coordinate
    /// `lane_component` decodes on the flat walk.
    pub(crate) fn lane_index_exact(&self) -> bool {
        self.reduce.len() == 1
            || self.reduce_extents[self.reduce_extents.len() - 1].is_multiple_of(self.lw)
    }
}

/// How many of the accumulator's innermost scalars one block column holds: one at a folded step,
/// `spread` where a wide rhs line spans several sink cells, and the rhs's own line otherwise.
///
/// `nr` counts in these and so do the extents `col` unravels over, so the rule is stated once
/// here rather than spelled out at each of them.
fn column_cell_width(contracted_per_step: usize, spread: usize, vw: usize) -> usize {
    if contracted_per_step > 1 {
        1
    } else if spread > 1 {
        spread
    } else {
        vw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_space;

    const B: Axis = Axis(0);
    const OH: Axis = Axis(1);
    const OW: Axis = Axis(2);
    const C: Axis = Axis(3);
    const RH: Axis = Axis(4);
    const RW: Axis = Axis(5);

    /// A depthwise-shaped contraction: the filter shares only the channel with the accumulator, so
    /// `MatrixAxes::accumulator` stops the column group at the top and leaves `out_h`, `out_w` and
    /// the channel all in it.
    ///
    /// The gather nest resolves an operand's read by `acc.position(axis)`, so the coordinate it
    /// assembles has to carry one entry per axis of the accumulator's space. It used to carry
    /// `batch… , row, col` regardless, which is one entry short per extra column axis and read
    /// past the end of its own list.
    #[test]
    fn the_cell_coordinate_covers_every_accumulator_axis() {
        let acc = flat_space(&[(B, 1), (OH, 1), (OW, 4), (C, 4)]);
        let lhs = flat_space(&[(C, 4), (RH, 3), (RW, 3)]);
        let rhs = flat_space(&[(B, 1), (OH, 1), (OW, 4), (C, 4), (RH, 3), (RW, 3)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.acc_axes.col_split, 1);
        assert_eq!(
            shape.batch_extents().len()
                + shape.row_extents().len()
                + shape.column_line_extents().len(),
            acc.rank()
        );
        assert_eq!(shape.row_extents().iter().product::<usize>(), shape.mr);
        assert_eq!(
            shape.column_line_extents().iter().product::<usize>(),
            shape.nr
        );
    }

    /// The plain batched matmul the nest was written against: one axis per edge, so both unravels
    /// are the identity and the coordinate is `batch…, row, col` as before.
    #[test]
    fn a_single_axis_per_edge_leaves_the_coordinate_unchanged() {
        let acc = flat_space(&[(B, 2), (OH, 4), (C, 8)]);
        let lhs = flat_space(&[(B, 2), (OH, 4), (RH, 6)]);
        let rhs = flat_space(&[(B, 2), (RH, 6), (C, 8)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.batch_extents(), vec![2]);
        assert_eq!(shape.row_extents(), vec![4]);
        assert_eq!(shape.column_line_extents(), vec![2]);
    }
}
