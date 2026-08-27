//! The 2-D matrix views over a [`Tile`]. Two [`Layout`]s re-view the tile's N-D [`Space`] as a
//! plain [`Coords2d`] `(row, col)` matrix: [`TileMatrix`] pins a batch prefix and exposes a
//! row group and a column group, each spanning as many axes as its edge needs. [`Tile::matrix`] and
//! friends then wrap the result as a [`MatrixView`] (a [`MaskedView`] carrying the comptime
//! overhang-`check` flag). Used by the matmul leaves and [`copy_2d()`].

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 2-D ([`TileMatrix`]) view: one matrix of a [`Tile`].
pub type MatrixView<'a, T> = MaskedView<'a, T, Coords2d>;
/// The mutable twin of [`MatrixView`].
pub type MatrixViewMut<'a, T> = MaskedViewMut<'a, T, Coords2d>;

/// A [`Layout`] presenting a tile's logical box as one `(row, col)` matrix over three axes of
/// axes, in the space's own order: a *batch* prefix already pinned to one matrix, then the axes
/// `row` unravels over, then the axes `col` unravels over (the innermost a line count).
///
/// One type because there is one concept. A plain batched matmul pins its leading axes and exposes
/// exactly two, so both axes hold one axis and the unravels are the identity. A convolution
/// contracts over its taps *and* its channels, which no pinning exposes as one edge, so its `k`
/// group holds several. A [partitioned](Composition::Disjoint) axis is the same again: an operand
/// spanning `(M, KB, KI)` reads as `M·KB` rows by `KI` columns, and the block index rides in the
/// row group at extent `1`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TileMatrix {
    /// Leading coordinates, already resolved to this matrix.
    batches: Coords<u32>,
    /// The group `row` unravels over, in the space's axis order.
    row_extents: Coords<u32>,
    /// The group `col` unravels over, the innermost a line count.
    col_extents: Coords<u32>,
    tile_shape: Coords2d,
}

/// A [`TileMatrix`] over the operand's mapping: what every 2-D reader of a tile sees, from the
/// matmul leaves to an mma fragment's stage.
pub type ProjectedMatrix = Projected<TileMatrix>;

#[cube]
impl TileMatrix {
    pub fn new(
        batches: Coords<u32>,
        row_extents: Coords<u32>,
        col_extents: Coords<u32>,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> Self {
        TileMatrix {
            batches,
            row_extents,
            col_extents,
            tile_shape: (rows as u32, cols as u32).runtime(),
        }
    }
}

#[cube]
impl Layout for TileMatrix {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        concat3(
            &self.batches,
            &unravel(&self.row_extents, row),
            &unravel(&self.col_extents, col),
        )
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        within_2d(pos, self.tile_shape)
    }
}

/// Which of a tile's axes form the matrix a 2-D reader sees: a batch prefix pinned to one matrix,
/// then the group `row` unravels over, then the group `col` unravels over.
///
/// Stated rather than assumed, because the axes alone cannot say. `(M, KB, KI)` with the block
/// index pinned to one block is a `M x KI` matrix, and `(B, M, K)` is a batch of `M x K` ones;
/// both are rank 3. A caller that knows the matrix it wants says so, and a grouping that is not a
/// face of the tile's box is refused here rather than read out of bounds.
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
    pub fn trailing_pair(space: &Space) -> Self {
        let rank = space.rank();
        MatrixAxes {
            row_split: rank - 2,
            col_split: rank - 1,
        }
    }

    /// The axes giving a `rows x cols` matrix, both scalar, found from the innermost axis
    /// outwards. An empty row group is legal exactly when `rows` is `1`: the row coordinate is
    /// then always `0` and the axes above sit in the batch prefix, which pins them the same way.
    ///
    /// `None` where no grouping of this tile's axes is that matrix, which is the question "does a
    /// 2-D reading describe this operand at all": a contraction the operand does not carry as one
    /// run of axes has no `k` edge, and is read a cell at a time instead.
    pub fn find(space: &Space, rows: usize, cols: usize) -> Option<Self> {
        let rank = space.rank();
        let mut col_split = rank;
        let mut trailing = 1;
        while col_split > 0 && trailing < cols {
            col_split -= 1;
            trailing *= space.extent_at(col_split);
        }
        if trailing != cols {
            return None;
        }
        let mut row_split = col_split;
        let mut middle = 1;
        while row_split > 0 && middle < rows {
            row_split -= 1;
            middle *= space.extent_at(row_split);
        }
        if middle != rows {
            return None;
        }
        // A degenerate leading axis multiplies nothing, so it belongs to the row group rather than
        // to a batch prefix that would pin it to the same `0`. Absorbing it keeps one answer per
        // question: without this a rank-3 box with a `1` on top axes two ways.
        while row_split > 0 && space.extent_at(row_split - 1) == 1 {
            row_split -= 1;
        }
        Some(MatrixAxes {
            row_split,
            col_split,
        })
    }

    /// [`of`](Self::of) over a tile's *whole* box, no batch prefix: every axis lands in one group
    /// or the other, and the column group holds the innermost (vectorized) axis whole. What an mma
    /// fragment reads, where `cols` is stated in scalars and the view serves it as lines.
    pub fn whole(space: &Space, rows: usize, cols: usize, vector_size: usize) -> Self {
        let rank = space.rank();
        let axes = MatrixAxes::of(space, rows, cols);
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

    /// [`find`](Self::find) where the caller has already established that the matrix exists.
    pub fn of(space: &Space, rows: usize, cols: usize) -> Self {
        MatrixAxes::find(space, rows, cols).unwrap_or_else(|| {
            let extents = (0..space.rank())
                .map(|p| space.extent_at(p))
                .collect::<Vec<_>>();
            panic!(
                "MatrixAxes: no grouping of this tile's axes gives a {rows}x{cols} matrix (its \
                 extents are {extents:?})"
            )
        })
    }
}

/// The leading (batch) extents a matrix index unravels over, in the space's axis order.
///
/// A direct operand reads them off the window, which is the only place a
/// [`Dynamic`](crate::Extent::Dynamic) top-level axis carries a size. A gathered one reads them
/// off the space: its window is boxed in *physical* axes, which are fewer than the logical ones
/// and are combinations of them, so no entry of it sizes a logical axis.
#[cube]
fn leading_extents(
    bound: &Coords<u32>,
    #[comptime] space: &Space,
    #[comptime] gathered: bool,
    #[comptime] upto: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..upto {
        if comptime!(gathered) {
            out.push(comptime!(space.extent_at(p) as u32));
        } else {
            out.push(bound.at(p));
        }
    }

    out
}

/// The `i`-th matrix of a tile whose window is `bound`, read over the axes [`MatrixAxes`] names:
/// the batch prefix pinned to `i` unraveled over its extents, the other two edges exposed as
/// `row` and `col`. The column edge is a line count, so the innermost extent divides by the width.
#[cube]
pub(crate) fn batch_matrix(
    bound: &Coords<u32>,
    #[comptime] space: &Space,
    #[comptime] gathered: bool,
    #[comptime] vector_size: usize,
    #[comptime] axes: MatrixAxes,
    i: usize,
) -> TileMatrix {
    let rank = comptime!(space.rank());
    let rows = comptime!(
        (axes.row_split..axes.col_split)
            .map(|p| space.extent_at(p))
            .product::<usize>()
    );
    // Rounded up like the buffer's own line count (`storage_extents`): a padded stage's innermost
    // extent need not fill whole lines, and the box a checked read tests against has to include
    // the partial last one it really holds. `cols` is a shape here, never a stride, so this only
    // widens the bound.
    let cols = comptime!(
        line_extents(space, vector_size, axes.col_split, rank)
            .iter()
            .product::<usize>()
    );
    let extents = leading_extents(bound, comptime!(space), gathered, comptime!(axes.row_split));

    TileMatrix::new(
        unravel(&extents, i.fcast::<u32>()),
        const_coords(comptime!(line_extents(
            space,
            vector_size,
            axes.row_split,
            axes.col_split
        ))),
        const_coords(comptime!(line_extents(
            space,
            vector_size,
            axes.col_split,
            rank
        ))),
        rows,
        cols,
    )
}

/// The tile's whole logical box as one `rows x cols` matrix, its axes grouped by
/// [`MatrixAxes::whole`]. `cols` is scalar, as a fragment states it; the view serves lines, so
/// the column edge and the innermost extent both divide by the width.
#[cube]
pub(crate) fn whole_matrix(
    #[comptime] space: &Space,
    #[comptime] vector_size: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
) -> TileMatrix {
    let rank = comptime!(space.rank());
    let split = comptime!(MatrixAxes::whole(space, rows, cols, vector_size).col_split);

    TileMatrix::new(
        Coords::<u32>::new(),
        const_coords(comptime!(line_extents(space, vector_size, 0, split))),
        const_coords(comptime!(line_extents(space, vector_size, split, rank))),
        rows,
        comptime!(cols / vector_size),
    )
}

/// [`batch_matrix`] over the operand's mapping: the `i`-th batch matrix as every 2-D reader of a
/// tile sees it.
#[cube]
pub(crate) fn projected_batch_matrix(
    bound: &Coords<u32>,
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
    #[comptime] axes: MatrixAxes,
    i: usize,
) -> ProjectedMatrix {
    // A partition is not a gather: its windows tile, so the window still sizes every logical axis.
    let gathered = comptime!(projection.composition() == Composition::Overlapping);
    ProjectedMatrix::new(
        batch_matrix(bound, comptime!(&space), gathered, vector_size, axes, i),
        axis_projection(comptime!(space), comptime!(projection), map, vector_size),
    )
}

/// [`whole_matrix`] over the operand's mapping: the whole logical box as the `rows x cols`
/// matrix an mma fragment reads.
#[cube]
pub(crate) fn projected_whole_matrix(
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
) -> ProjectedMatrix {
    ProjectedMatrix::new(
        whole_matrix(comptime!(&space), vector_size, rows, cols),
        axis_projection(comptime!(space), comptime!(projection), map, vector_size),
    )
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// The `i`-th batch matrix over `Vector<T, W>` lines (`W` = [`width`](Tile::width)).
    pub fn matrix<W: Size>(&self, i: usize) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let bound = g.extent();
                let layout = projected_batch_matrix(
                    &bound,
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    comptime!(MatrixAxes::trailing_pair(&self.space)),
                    i,
                );
                g.masked::<W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
            TileKind::Procedural(_) => panic!("Tile::matrix: a procedural tile has no memory view"),
        }
    }

    /// Mutable version of [`matrix`](Tile::matrix). Only supported for direct projections.
    pub fn matrix_mut<W: Size>(&mut self, i: usize) -> MatrixViewMut<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                comptime!(assert!(
                    g.projection.is_direct(),
                    "Tile::matrix_mut: a gathered operand aliases under a write"
                ));
                let bound = g.extent();
                let layout = projected_batch_matrix(
                    &bound,
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    comptime!(MatrixAxes::trailing_pair(&self.space)),
                    i,
                );
                g.masked_mut::<W, Coords2d, ProjectedMatrix>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix_mut: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix_mut: a tma source has no element view"),
            TileKind::Procedural(_) => {
                panic!("Tile::matrix_mut: a procedural tile is not writable")
            }
        }
    }

    /// The `i`-th batch matrix, read through whatever [`Packing`] this tile carries.
    ///
    /// The one place a packing becomes a storage element: every leaf used to re-derive the
    /// `i8`/`u32` choice from a bare factor, and the return type never mentions it.
    pub fn matrix_packed<W: Size>(
        &self,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let served = self.vector_size();
        let packing = self.packing();
        let physical = comptime!(packing.physical(served));
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.matrix_transparent::<T, WP, W>(axes, i)
            }
            Packing::Native => {
                let size!(WP) = physical;
                self.matrix_transparent::<i8, WP, W>(axes, i)
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = physical;
                self.matrix_transparent::<u32, WP, W>(axes, i)
            }
        }
    }

    /// [`matrix_packed`](Tile::matrix_packed) with a scales operand folded into every read.
    ///
    /// The scales are read at the same matrix coordinate, through their own axes, so the caller
    /// states each matrix once and the multiply happens under the view. Whatever consumes this
    /// consumes an ordinary [`MatrixView`]: a contraction over a scaled operand *is* the plain
    /// contraction.
    pub fn matrix_scaled<'a, W: Size, S: Numeric, SW: Size>(
        &'a self,
        #[comptime] axes: MatrixAxes,
        scales: &'a Tile<S>,
        #[comptime] scale_axes: MatrixAxes,
        i: usize,
    ) -> MatrixView<'a, Vector<T, W>> {
        let width = self.vector_size();
        let values = self.matrix_packed::<W>(axes, i);
        let scale_lines = scales.matrix_packed::<SW>(scale_axes, i);
        // A scale read past the values' bound is never used: the values mask first, and a masked
        // value is zero whatever it is multiplied by.
        let check = comptime!(values.check);
        let scaled =
            ScaledView::<T, W, S, SW>::new(values.into_view(), scale_lines.into_view(), width);
        MaskedView::new(scaled.view(), check)
    }

    /// [`matrix_packed`](Tile::matrix_packed) at a stated storage element `I` and physical line
    /// `WP`: a plain tile is read as it stands, a quantized one dequantizes each `(row, col)` per
    /// its scheme, with no dequantize-into-`f32` fill.
    pub fn matrix_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let bound = g.extent();
                let layout = projected_batch_matrix(
                    &bound,
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    axes,
                    i,
                );
                g.matrix_transparent::<I, WP, W, ProjectedMatrix>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix_transparent: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => {
                panic!("Tile::matrix_transparent: a tma source has no element view")
            }
            TileKind::Procedural(_) => {
                panic!("Tile::matrix_transparent: a procedural tile has no memory view")
            }
        }
    }

    /// The fragment's grouped matrix, read through whatever [`Packing`] this tile carries. The
    /// manual-mma twin of [`matrix_packed`](Tile::matrix_packed).
    ///
    /// `cols` is scalar, as an mma definition states it; the view serves lines.
    pub fn fragment_matrix_packed<W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let served = self.vector_size();
        let packing = self.packing();
        let physical = comptime!(packing.physical(served));
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.fragment_matrix::<T, WP, W>(rows, cols)
            }
            Packing::Native => {
                let size!(WP) = physical;
                self.fragment_matrix::<i8, WP, W>(rows, cols)
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = physical;
                self.fragment_matrix::<u32, WP, W>(rows, cols)
            }
        }
    }

    /// [`fragment_matrix_packed`](Tile::fragment_matrix_packed) at a stated storage element:
    /// several logical axes may flatten into one edge, so an operand whose contraction spans its
    /// taps *and* its channels still has a `k` edge, and a gathered one reads it straight out of
    /// its compacted stage.
    pub fn fragment_matrix<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let layout = projected_whole_matrix(
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    rows,
                    cols,
                );
                g.matrix_transparent::<I, WP, W, ProjectedMatrix>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::fragment_matrix: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => {
                panic!("Tile::fragment_matrix: a tma source has no element view")
            }
            TileKind::Procedural(_) => {
                panic!("Tile::fragment_matrix: a procedural tile has no memory view")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OH: Axis = Axis(0);
    const RH: Axis = Axis(1);
    const CI: Axis = Axis(2);

    /// A plain matmul operand: one axis per edge, the split right down the middle.
    #[test]
    fn a_rank_two_operand_splits_between_its_axes() {
        let s = flat_space(&[(OH, 8), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 4, 1).col_split, 1);
    }

    /// The convolution shape: the trailing *two* axes are the contraction, so `k` is their
    /// product and the split leaves only the output axis in the row group.
    #[test]
    fn a_contraction_over_two_axes_splits_before_both() {
        let s = flat_space(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 8, 1).col_split, 1);
    }

    /// Both edges spanning several axes, which is what a 2-D convolution's input needs.
    #[test]
    fn both_edges_may_span_several_axes() {
        let s = flat_space(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(MatrixAxes::whole(&s, 12, 2, 2).col_split, 2);
    }

    /// The smallest column group that reaches `cols` wins, so a degenerate leading axis stays in
    /// the row group rather than being swept into the column one. Either split addresses the same
    /// cells; taking the smaller keeps the answer deterministic.
    #[test]
    fn a_degenerate_leading_axis_stays_in_the_row_group() {
        let s = flat_space(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 1, 8, 1).col_split, 1);
    }

    /// No split gives the asked-for edges: the extents multiply to 32, and 8x8 is not a face.
    #[test]
    #[should_panic(expected = "no grouping of this tile's axes")]
    fn a_mismatched_fragment_is_refused() {
        let s = flat_space(&[(OH, 8), (RH, 2), (CI, 2)]);
        MatrixAxes::whole(&s, 8, 8, 1);
    }

    /// The column group must contain the innermost axis: the view serves lines along `col`, so a
    /// split that leaves the vectorized axis in the row group has no line to read.
    #[test]
    #[should_panic(expected = "innermost (vectorized) axis")]
    fn the_vectorized_axis_must_land_in_the_column_group() {
        let s = flat_space(&[(OH, 8), (CI, 4)]);
        MatrixAxes::whole(&s, 32, 1, 1);
    }

    /// The column edge and the innermost extent are both counted in lines, so a width that does
    /// not divide the innermost extent would scale the two by different amounts.
    #[test]
    #[should_panic(expected = "whole number of 4-wide lines")]
    fn a_partial_innermost_line_is_refused() {
        let s = flat_space(&[(OH, 8), (CI, 6)]);
        MatrixAxes::whole(&s, 8, 6, 4);
    }
}
