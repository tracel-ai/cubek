//! Which of a space's axes form the matrix a 2-D reader sees.

use core::fmt::{self, Display, Formatter};

use crate::{Extent, Space};

/// Which of a tile's axes form the matrix a 2-D reader sees: a batch prefix pinned to one matrix,
/// then the group `row` unravels over, then the group `col` unravels over.
///
/// Stated rather than assumed, because the axes alone cannot say: `(M, KB, KI)` pinned to one
/// block is an `M x KI` matrix and `(B, M, K)` a batch of `M x K` ones, both rank 3. A grouping
/// that is not a face of the tile's box is refused here rather than read out of bounds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixAxes {
    /// Where the row group starts; everything before it is batch.
    pub row_split: usize,
    /// Where the column group starts.
    pub col_split: usize,
}

impl MatrixAxes {
    /// The trailing pair: leading axes batch, the last two the matrix. What a tile whose axes are
    /// already `[batch…, row, col]` reads through, which is every operand of an unpartitioned
    /// problem.
    pub(crate) fn trailing(space: &Space) -> Self {
        let rank = space.rank();
        MatrixAxes {
            row_split: rank - 2,
            col_split: rank - 1,
        }
    }

    /// The two edges a matrix reader takes as its rows and columns: the innermost axis, and the
    /// last axis above it of extent past one.
    ///
    /// An axis of extent one folds away, so a split contraction's block digit or a column tile's
    /// index does not stand between a fragment and its rows. What a fragment, a partition and a
    /// trailing region read through, none of which knows a shape to [`find`](Self::find) one from.
    pub fn edges(space: &Space) -> Self {
        let rank = space.rank();
        let col_split = rank - 1;
        // A dynamic axis is not a number one: its size is the launch's, and it is a row edge.
        let row_split = (0..col_split)
            .rev()
            .find(|&p| !matches!(space.extent_raw(space.axis_at(p)), Extent::Static(1)))
            .unwrap_or(col_split.saturating_sub(1));
        MatrixAxes {
            row_split,
            col_split,
        }
    }

    /// An accumulator's matrix, against the lhs it is contracted with.
    ///
    /// The innermost axis is a column edge by construction (the sink lines along it); the group
    /// reaches up to the first axis the lhs spans, which must be walked against the lhs rather than
    /// folded into a column. The row edge is the axis before the group; anything above is batch.
    ///
    /// This is what lets a `[bm, bn]` scheme split `N` into a block index and a position inside
    /// it: both are the rhs's alone, so both are columns, where taking the last axis alone would
    /// have made the block index a row.
    pub fn accumulator(acc: &Space, lhs: &Space) -> Self {
        let mut col_split = acc.rank() - 1;
        while col_split > 1 && !lhs.contains(acc.axis_at(col_split - 1)) {
            col_split -= 1;
        }
        MatrixAxes {
            row_split: col_split - 1,
            col_split,
        }
    }

    /// The row edge these axes give in `space`: the group between the batch prefix and the
    /// columns, multiplied out.
    pub fn rows(&self, space: &Space) -> usize {
        (self.row_split..self.col_split)
            .map(|p| space.extent_at(p))
            .product()
    }

    /// The column edge, in scalars.
    pub fn cols(&self, space: &Space) -> usize {
        (self.col_split..space.rank())
            .map(|p| space.extent_at(p))
            .product()
    }

    /// The axes giving a `rows x cols` matrix, both scalar, found from the innermost axis
    /// outwards. An empty row group is legal exactly when `rows` is `1`: the row coordinate is
    /// then always `0` and the axes above sit in the batch prefix, which pins them the same way.
    ///
    /// Refused where no grouping of this tile's axes is that matrix, which is the question "does a
    /// 2-D reading describe this operand at all": a contraction the operand does not carry as one
    /// run of axes has no `k` edge, and is read a cell at a time instead.
    pub fn new(space: &Space, rows: usize, cols: usize) -> Result<Self, NoMatrix> {
        let rank = space.rank();
        let mut col_split = rank;
        let mut trailing = 1;
        while col_split > 0 && trailing < cols {
            col_split -= 1;
            trailing *= space.extent_at(col_split);
        }
        if trailing != cols {
            return Err(NoMatrix::new(space, rows, cols));
        }
        let mut row_split = col_split;
        let mut middle = 1;
        while row_split > 0 && middle < rows {
            row_split -= 1;
            middle *= space.extent_at(row_split);
        }
        if middle != rows {
            return Err(NoMatrix::new(space, rows, cols));
        }
        // A degenerate leading axis multiplies nothing, so it belongs to the row group rather than
        // to a batch prefix that would pin it to the same `0`. Absorbing it keeps one answer per
        // question: without this a rank-3 box with a `1` on top axes two ways.
        while row_split > 0 && space.extent_at(row_split - 1) == 1 {
            row_split -= 1;
        }
        Ok(MatrixAxes {
            row_split,
            col_split,
        })
    }

    /// [`new`](Self::new) over a tile's *whole* box, no batch prefix: every axis lands in one group
    /// or the other, and the column group holds the innermost (vectorized) axis whole. What an mma
    /// fragment reads, where `cols` is stated in scalars and the view serves it as lines.
    pub fn whole(space: &Space, rows: usize, cols: usize, vector_size: usize) -> Self {
        let rank = space.rank();
        let axes = MatrixAxes::new(space, rows, cols).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            axes.row_split == 0,
            "MatrixAxes::whole: this tile has axes above its {rows} rows, so its box is a \
             batch of matrices rather than one"
        );
        // The view serves lines along `col`, so the vectorized axis has to land in the column
        // group, whole: the column edge and the innermost extent are both counted in lines, and a
        // partial line would divide one of them to a different group product.
        assert!(
            axes.col_split < rank,
            "MatrixAxes::whole: the innermost (vectorized) axis must be part of the column group"
        );
        let innermost = space.extent_at(rank - 1);
        assert!(
            innermost.is_multiple_of(vector_size),
            "MatrixAxes::whole: the innermost extent {innermost} is not a whole number of \
             {vector_size}-wide lines"
        );
        axes
    }
}

/// Why a space has no `rows x cols` reading: no grouping of its axes multiplies out to it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NoMatrix {
    pub rows: usize,
    pub cols: usize,
    pub extents: Vec<usize>,
}

impl NoMatrix {
    fn new(space: &Space, rows: usize, cols: usize) -> Self {
        NoMatrix {
            rows,
            cols,
            extents: (0..space.rank()).map(|p| space.extent_at(p)).collect(),
        }
    }
}

impl Display for NoMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatrixAxes: no grouping of this tile's axes gives a {}x{} matrix (its extents are \
             {:?})",
            self.rows, self.cols, self.extents
        )
    }
}

impl std::error::Error for NoMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axis;

    const OH: Axis = Axis(0);
    const RH: Axis = Axis(1);
    const CI: Axis = Axis(2);

    /// A plain matmul operand: one axis per edge, the split right down the middle.
    #[test]
    fn a_rank_two_operand_splits_between_its_axes() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 4, 1).col_split, 1);
    }

    /// The convolution shape: the trailing *two* axes are the contraction, so `k` is their
    /// product and the split leaves only the output axis in the row group.
    #[test]
    fn a_contraction_over_two_axes_splits_before_both() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 8, 1).col_split, 1);
    }

    /// Both edges spanning several axes, which is what a 2-D convolution's input needs.
    #[test]
    fn both_edges_may_span_several_axes() {
        let s = Space::new(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(MatrixAxes::whole(&s, 12, 2, 2).col_split, 2);
    }

    /// The smallest column group that reaches `cols` wins, so a degenerate leading axis stays in
    /// the row group rather than being swept into the column one. Either split addresses the same
    /// cells; taking the smaller keeps the answer deterministic.
    #[test]
    fn a_degenerate_leading_axis_stays_in_the_row_group() {
        let s = Space::new(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 1, 8, 1).col_split, 1);
    }

    /// No split gives the asked-for edges: the extents multiply to 32, and 8x8 is not a face.
    #[test]
    #[should_panic(expected = "no grouping of this tile's axes")]
    fn a_mismatched_fragment_is_refused() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 2)]);
        MatrixAxes::whole(&s, 8, 8, 1);
    }

    /// The column group must contain the innermost axis: the view serves lines along `col`, so a
    /// split that leaves the vectorized axis in the row group has no line to read.
    #[test]
    #[should_panic(expected = "innermost (vectorized) axis")]
    fn the_vectorized_axis_must_land_in_the_column_group() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        MatrixAxes::whole(&s, 32, 1, 1);
    }

    /// The column edge and the innermost extent are both counted in lines, so a width that does
    /// not divide the innermost extent would scale the two by different amounts.
    #[test]
    #[should_panic(expected = "whole number of 4-wide lines")]
    fn a_partial_innermost_line_is_refused() {
        let s = Space::new(&[(OH, 8), (CI, 6)]);
        MatrixAxes::whole(&s, 8, 6, 4);
    }
}
