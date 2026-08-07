//! The 2-D matrix view over a [`Tile`]. [`BatchMatrix`] is a [`Layout`] that re-views the tile's
//! N-D [`Space`] as a plain [`Coords2d`] `(row, col)` matrix: leading batch axes pinned, trailing
//! two exposed; [`Tile::matrix`]/[`Tile::matrix_mut`] then wrap it as a [`MatrixView`]/
//! [`MatrixViewMut`] (a [`MaskedView`] carrying the comptime overhang-`check` flag). Used by the
//! matmul leaves and [`copy_2d()`].
//!
//! [`ProjectedBatchMatrix`] is the same surface over a *gathered* operand, whose logical axes
//! outnumber its buffer's physical ones: [`BatchMatrix`] resolves the matrix coordinate to the
//! tile's logical one, then [`AxisProjection`] folds that onto the window's physical one. Every
//! generalized step is gated on [`Projection::is_direct`], so a direct operand keeps its exact
//! previous codegen.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 2-D ([`BatchMatrix`]) view: one batch matrix of a [`Tile`].
pub type MatrixView<'a, T> = MaskedView<'a, T, Coords2d>;
/// The mutable twin of [`MatrixView`].
pub type MatrixViewMut<'a, T> = MaskedViewMut<'a, T, Coords2d>;

/// A [`Layout`] mapping a matrix coordinate `(row, col)` to the tile's source
/// coordinate `[batches…, row, col]`: leading batch axes pinned, trailing two exposed.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct BatchMatrix {
    /// [`Coords`], not [`CoordsDyn`]: the pinned coordinates are never reassigned, and a
    /// `Sequence` holder copies them into mutable slots, which erases the constness the fold
    /// arithmetic downstream needs (see [`Coords`](crate::Coords)).
    batches: Coords<u32>,
    tile_shape: Coords2d,
}

#[cube]
impl BatchMatrix {
    pub fn new(batches: Coords<u32>, #[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        BatchMatrix {
            batches,
            tile_shape: (rows as u32, cols as u32).runtime(),
        }
    }
}

#[cube]
impl Layout for BatchMatrix {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        // Built up rather than cloned-and-extended: an empty `Sequence` has nothing to copy into
        // mutable slots, so each pinned coordinate reaches the source position as it was folded.
        let mut out = CoordsDyn::new();
        #[unroll]
        for p in 0..self.batches.len() {
            out.push(self.batches.at(p));
        }
        out.push(t0);
        out.push(t1);
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (t0, t1) = pos;
        let (s0, s1) = self.tile_shape;
        t0 < s0 && t1 < s1
    }
}

/// [`BatchMatrix`] over a *gathered* operand.
pub type ProjectedBatchMatrix = Projected<BatchMatrix>;

/// [`GroupedMatrix`] over a *gathered* operand: what an mma fragment reads a compacted stage
/// through.
pub type ProjectedGroupedMatrix = Projected<GroupedMatrix>;

/// A [`Layout`] presenting the tile's *whole* logical box as one `(row, col)` matrix: `row`
/// unravels over a leading group of logical axes, `col` over the trailing group (the innermost a
/// line count). Where [`BatchMatrix`] pins the leading axes and exposes exactly two, this
/// flattens them into the two edges.
///
/// That is the difference between a tile that has a matrix face and one that *is* a matrix. An
/// mma fragment contracts over a single `k` edge, but an operand's contraction can span several
/// logical axes (a convolution contracts over its taps *and* its channels), and no pinning
/// exposes that as one edge. Splitting the axes into two groups does, and the split is where the
/// contraction starts: `[M…, K…]` for the `A` role, `[K…, N…]` for `B`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct GroupedMatrix {
    /// Leading group, in the space's axis order.
    row_extents: Coords<u32>,
    /// Trailing group, the innermost a line count.
    col_extents: Coords<u32>,
    tile_shape: Coords2d,
}

#[cube]
impl GroupedMatrix {
    pub fn new(
        row_extents: Coords<u32>,
        col_extents: Coords<u32>,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> Self {
        GroupedMatrix {
            row_extents,
            col_extents,
            tile_shape: (rows as u32, cols as u32).runtime(),
        }
    }
}

#[cube]
impl Layout for GroupedMatrix {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        let rows = unravel(&self.row_extents, row);
        let cols = unravel(&self.col_extents, col);

        let mut out = CoordsDyn::new();
        #[unroll]
        for p in 0..rows.len() {
            out.push(rows.at(p));
        }
        #[unroll]
        for p in 0..cols.len() {
            out.push(cols.at(p));
        }
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (row, col) = pos;
        let (s0, s1) = self.tile_shape;
        row < s0 && col < s1
    }
}

/// The leading (batch) extents a matrix index unravels over, in the space's axis order.
///
/// A direct operand reads them off the window, which is the only place a
/// [`Dynamic`](crate::Extent::Dynamic) top-level axis carries a size. A gathered one reads them
/// off the space: its window is boxed in *physical* axes, which are fewer than the logical ones
/// and are combinations of them, so no entry of it sizes a logical axis. Those extents are always
/// comptime, the same ones [`axis_projection`] shapes its logical box with.
#[cube]
fn leading_extents(
    bound: &Coords<u32>,
    #[comptime] space: Space,
    #[comptime] gathered: bool,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..comptime!(space.rank() - 2) {
        if comptime!(gathered) {
            out.push(comptime!(space.extent_at(p) as u32));
        } else {
            out.push(bound.at(p));
        }
    }

    out
}

/// Unravel a flat index over `extents`: `out[p] = (i / Π extents[p+1..]) % extents[p]`. Folding
/// throughout, so comptime extents and a comptime index leave no arithmetic behind.
///
/// The outermost digit keeps the bare quotient: it has no enclosing block to be reduced against,
/// and `i` is always within the product (a matrix index counts the matrices, a row counts the
/// rows), so the modulo there is the identity. Same reasoning as
/// [`Projection::digit`](crate::Projection::digit), whose outermost fragment carries `None`.
#[cube]
fn unravel(extents: &Coords<u32>, i: u32) -> Coords<u32> {
    let n = extents.len();
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..n {
        let digit = i.fdiv(extents.fproduct(comptime!(((p + 1)..n).collect::<Vec<_>>())));
        if comptime!(p == 0) {
            out.push(digit);
        } else {
            out.push(digit.frem(extents.at(p)));
        }
    }

    out
}

/// The `i`-th batch matrix of a tile whose window is `bound`: leading axes pinned to `i`
/// unraveled over their extents, trailing two exposed. `cols` is a line count, so the innermost
/// extent divides by the width.
#[cube]
pub(crate) fn batch_matrix(
    bound: &Coords<u32>,
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    #[comptime] vector_size: usize,
    i: usize,
) -> BatchMatrix {
    let rank = comptime!(space.rank());
    let rows = comptime!(space.extent_at(rank - 2));
    let cols = comptime!(space.extent_at(rank - 1) / vector_size);
    let extents = leading_extents(bound, comptime!(space), comptime!(!projection.is_direct()));

    BatchMatrix::new(unravel(&extents, i.fcast::<u32>()), rows, cols)
}

/// Where the space's axes split into a leading group whose extents multiply to `rows` and a
/// trailing group whose extents multiply to `cols`, both scalar: the position the trailing group
/// starts at.
///
/// The two edges pin the split, so a caller states the matrix it wants rather than how the tile's
/// axes are grouped into it. A `(rows, cols)` pair that is not a face of this box is a mismatched
/// fragment, caught here rather than read out of bounds.
pub(crate) fn matrix_split(space: &Space, rows: usize, cols: usize) -> usize {
    let rank = space.rank();
    let mut split = rank;
    let mut trailing = 1;
    while split > 0 && trailing < cols {
        split -= 1;
        trailing *= space.extent_at(split);
    }
    let leading: usize = (0..split).map(|p| space.extent_at(p)).product();
    assert!(
        trailing == cols && leading == rows,
        "matrix_split: no split of this tile's axes gives a {rows}x{cols} matrix (the trailing \
         axes multiply to {trailing}, the leading ones to {leading})"
    );
    // The innermost axis is the vectorized one, and the view serves lines along `col`, so it has
    // to land in the trailing group.
    assert!(
        split < rank,
        "matrix_split: the innermost (vectorized) axis must be part of the column group"
    );
    split
}

/// The tile's whole logical box as one `rows x cols` matrix, its axes split by
/// [`matrix_split`]. `cols` is scalar, as a fragment states it; the view serves lines, so the
/// column edge and the innermost extent both divide by the width.
#[cube]
pub(crate) fn grouped_matrix(
    #[comptime] space: Space,
    #[comptime] vector_size: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
) -> GroupedMatrix {
    let rank = comptime!(space.rank());
    let split = comptime!(matrix_split(&space, rows, cols));

    let mut row_extents = Coords::<u32>::new();
    #[unroll]
    for p in 0..split {
        row_extents.push(comptime!(space.extent_at(p) as u32));
    }

    let mut col_extents = Coords::<u32>::new();
    #[unroll]
    for p in split..rank {
        let e = comptime!(space.extent_at(p));
        col_extents.push(comptime!(
            (if p == rank - 1 { e / vector_size } else { e }) as u32
        ));
    }

    GroupedMatrix::new(
        row_extents,
        col_extents,
        rows,
        comptime!(cols / vector_size),
    )
}

/// [`batch_matrix`] with the operand's [`Projection`] applied under it: what a gathered operand's
/// 2-D readers go through.
#[cube]
fn projected_batch_matrix(
    bound: &Coords<u32>,
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    #[comptime] vector_size: usize,
    i: usize,
) -> ProjectedBatchMatrix {
    ProjectedBatchMatrix::new(
        batch_matrix(
            bound,
            comptime!(space.clone()),
            comptime!(projection.clone()),
            vector_size,
            i,
        ),
        axis_projection(comptime!(space), comptime!(projection), vector_size),
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
                let space = comptime!(self.space.clone());
                let projection = comptime!(g.projection.clone());
                if comptime!(projection.is_direct()) {
                    let layout = batch_matrix(&bound, space, projection, vector_size, i);
                    g.masked::<W, Coords2d, BatchMatrix>(layout)
                } else {
                    let layout = projected_batch_matrix(&bound, space, projection, vector_size, i);
                    g.masked::<W, Coords2d, ProjectedBatchMatrix>(layout)
                }
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
        }
    }

    pub fn matrix_mut<W: Size>(&mut self, i: usize) -> MatrixViewMut<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let bound = g.extent();
                let space = comptime!(self.space.clone());
                let projection = comptime!(g.projection.clone());
                if comptime!(projection.is_direct()) {
                    let layout = batch_matrix(&bound, space, projection, vector_size, i);
                    g.masked_mut::<W, Coords2d, BatchMatrix>(layout)
                } else {
                    let layout = projected_batch_matrix(&bound, space, projection, vector_size, i);
                    g.masked_mut::<W, Coords2d, ProjectedBatchMatrix>(layout)
                }
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix_mut: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix_mut: a tma source has no element view"),
        }
    }

    /// The `i`-th batch matrix as a quantization-transparent view: a plain tile is read as it
    /// stands, a quantized one dequantizes each `(row, col)` per its scheme (`I`/`WP` the
    /// storage element and physical line, as [`copy_from`](Tile::copy_from) recovers them). The
    /// dequant-at-read twin of [`matrix`](Tile::matrix); a leaf reads a quantized operand with no
    /// dequantize-into-`f32` fill.
    pub fn matrix_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        i: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let bound = g.extent();
                let space = comptime!(self.space.clone());
                let projection = comptime!(g.projection.clone());
                if comptime!(projection.is_direct()) {
                    let layout = batch_matrix(&bound, space, projection, vector_size, i);
                    g.matrix_transparent::<I, WP, W, BatchMatrix>(layout)
                } else {
                    let layout = projected_batch_matrix(&bound, space, projection, vector_size, i);
                    g.matrix_transparent::<I, WP, W, ProjectedBatchMatrix>(layout)
                }
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix_transparent: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => {
                panic!("Tile::matrix_transparent: a tma source has no element view")
            }
        }
    }

    /// This tile's whole logical box as the `rows x cols` matrix a fragment of that shape reads,
    /// quantization-transparent. The [`GroupedMatrix`] twin of
    /// [`matrix_transparent`](Tile::matrix_transparent): several logical axes may flatten into one
    /// edge, so an operand whose contraction spans its taps *and* its channels still has a `k`
    /// edge, and a gathered one reads it straight out of its compacted stage.
    ///
    /// `cols` is scalar, as an mma definition states it; the view serves lines.
    pub fn fragment_matrix<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let space = comptime!(self.space.clone());
                let projection = comptime!(g.projection.clone());
                let layout = grouped_matrix(comptime!(space.clone()), vector_size, rows, cols);
                if comptime!(projection.is_direct()) {
                    g.matrix_transparent::<I, WP, W, GroupedMatrix>(layout)
                } else {
                    let projected = ProjectedGroupedMatrix::new(
                        layout,
                        axis_projection(space, projection, vector_size),
                    );
                    g.matrix_transparent::<I, WP, W, ProjectedGroupedMatrix>(projected)
                }
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::fragment_matrix: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => {
                panic!("Tile::fragment_matrix: a tma source has no element view")
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

    fn space(extents: &[(Axis, usize)]) -> Space {
        let mut t = Tiling::new().extents(extents);
        t = t.level(WalkOrder::RowMajor, Schedule::Direct, |mut l| {
            for &(axis, e) in extents {
                l = l.axis(axis, Cut::sequential(e));
            }
            l
        });
        t.build()
    }

    /// A plain matmul operand: one axis per edge, the split right down the middle.
    #[test]
    fn a_rank_two_operand_splits_between_its_axes() {
        let s = space(&[(OH, 8), (CI, 4)]);
        assert_eq!(matrix_split(&s, 8, 4), 1);
    }

    /// The convolution shape: the trailing *two* axes are the contraction, so `k` is their
    /// product and the split leaves only the output axis in the row group.
    #[test]
    fn a_contraction_over_two_axes_splits_before_both() {
        let s = space(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(matrix_split(&s, 8, 8), 1);
    }

    /// Both edges spanning several axes, which is what a 2-D convolution's input needs.
    #[test]
    fn both_groups_may_span_several_axes() {
        let s = space(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(matrix_split(&s, 12, 2), 2);
    }

    /// The smallest column group that reaches `cols` wins, so a degenerate leading axis stays in
    /// the row group rather than being swept into the column one. Either split addresses the same
    /// cells; taking the smaller keeps the answer deterministic.
    #[test]
    fn a_degenerate_leading_axis_stays_in_the_row_group() {
        let s = space(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(matrix_split(&s, 1, 8), 1);
    }

    /// No split gives the asked-for edges: the extents multiply to 32, and 8x8 is not a face.
    #[test]
    #[should_panic(expected = "no split of this tile's axes")]
    fn a_mismatched_fragment_is_refused() {
        let s = space(&[(OH, 8), (RH, 2), (CI, 2)]);
        matrix_split(&s, 8, 8);
    }

    /// The column group must contain the innermost axis: the view serves lines along `col`, so a
    /// split that leaves the vectorized axis in the row group has no line to read.
    #[test]
    #[should_panic(expected = "innermost (vectorized) axis")]
    fn the_vectorized_axis_must_land_in_the_column_group() {
        let s = space(&[(OH, 8), (CI, 4)]);
        matrix_split(&s, 32, 1);
    }
}
