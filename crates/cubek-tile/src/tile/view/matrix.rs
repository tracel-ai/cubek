//! The 2-D matrix views over a [`Tile`]. Two [`Layout`]s re-view the tile's N-D [`Space`] as a
//! plain [`Coords2d`] `(row, col)` matrix: [`BatchMatrix`] pins the leading axes and exposes the
//! trailing two, [`GroupedMatrix`] flattens every axis into the two edges. [`Tile::matrix`] and
//! friends then wrap the result as a [`MatrixView`] (a [`MaskedView`] carrying the comptime
//! overhang-`check` flag). Used by the matmul leaves and [`copy_2d()`].

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
    /// Leading batch coordinates.
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
        let mut exposed = Coords::<u32>::new();
        exposed.push(t0);
        exposed.push(t1);
        concat(&self.batches, &exposed)
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

/// A 2-D matrix view over a projected tile with batch dimensions.
pub type ProjectedBatchMatrix = Projected<BatchMatrix>;

/// What an mma fragment reads its stage through: a [`GroupedMatrix`] over the operand's mapping.
pub type ProjectedGroupedMatrix = Projected<GroupedMatrix>;

/// A [`Layout`] presenting the tile's *whole* logical box as one `(row, col)` matrix: `row`
/// unravels over a leading group of logical axes, `col` over the trailing group (the innermost a
/// line count).
///
/// An mma fragment contracts over a single `k` edge, but an operand's contraction can span several
/// logical axes (a convolution contracts over its taps *and* its channels), which no pinning of
/// trailing axes exposes as one edge. The split is where the contraction starts: `[M…, K…]` for
/// the `A` role, `[K…, N…]` for `B`.
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
        concat(
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

/// The `i`-th batch matrix of a tile whose window is `bound`: leading axes pinned to `i`
/// unraveled over their extents, trailing two exposed. `cols` is a line count, so the innermost
/// extent divides by the width.
#[cube]
pub(crate) fn batch_matrix(
    bound: &Coords<u32>,
    #[comptime] space: &Space,
    #[comptime] gathered: bool,
    #[comptime] vector_size: usize,
    i: usize,
) -> BatchMatrix {
    let rank = comptime!(space.rank());
    let rows = comptime!(space.extent_at(rank - 2));
    let cols = comptime!(space.extent_at(rank - 1) / vector_size);
    let extents = leading_extents(bound, comptime!(space), gathered);

    BatchMatrix::new(unravel(&extents, i.fcast::<u32>()), rows, cols)
}

/// Where the space's axes split into a leading group whose extents multiply to `rows` and a
/// trailing group whose extents multiply to `cols`, both scalar: the position the trailing group
/// starts at.
///
/// The two edges pin the split, so a caller states the matrix it wants rather than how the tile's
/// axes are grouped into it. A `(rows, cols)` pair that is not a face of this box is a mismatched
/// fragment, caught here rather than read out of bounds.
pub(crate) fn matrix_split(space: &Space, rows: usize, cols: usize, vector_size: usize) -> usize {
    let rank = space.rank();
    let mut split = rank;
    let mut trailing = 1;
    // Find the split point that matches the column count.
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
    // to land in the trailing group, whole: the column edge and the innermost extent are both
    // counted in lines, and a partial line would divide one of them to a different group product.
    assert!(
        split < rank,
        "matrix_split: the innermost (vectorized) axis must be part of the column group"
    );
    let innermost = space.extent_at(rank - 1);
    assert!(
        innermost.is_multiple_of(vector_size),
        "matrix_split: the innermost extent {innermost} is not a whole number of {vector_size}-wide lines"
    );
    split
}

/// The tile's whole logical box as one `rows x cols` matrix, its axes split by
/// [`matrix_split`]. `cols` is scalar, as a fragment states it; the view serves lines, so the
/// column edge and the innermost extent both divide by the width.
#[cube]
pub(crate) fn grouped_matrix(
    #[comptime] space: &Space,
    #[comptime] vector_size: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
) -> GroupedMatrix {
    let rank = comptime!(space.rank());
    let split = comptime!(matrix_split(space, rows, cols, vector_size));

    GroupedMatrix::new(
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
    i: usize,
) -> ProjectedBatchMatrix {
    let gathered = comptime!(!projection.is_direct());
    ProjectedBatchMatrix::new(
        batch_matrix(bound, comptime!(&space), gathered, vector_size, i),
        axis_projection(comptime!(space), comptime!(projection), map, vector_size),
    )
}

/// [`grouped_matrix`] over the operand's mapping: the whole logical box as the `rows x cols`
/// matrix an mma fragment reads.
#[cube]
pub(crate) fn projected_grouped_matrix(
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
) -> ProjectedGroupedMatrix {
    ProjectedGroupedMatrix::new(
        grouped_matrix(comptime!(&space), vector_size, rows, cols),
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
                    i,
                );
                g.masked::<W, Coords2d, ProjectedBatchMatrix>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::matrix: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
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
                    i,
                );
                g.masked_mut::<W, Coords2d, ProjectedBatchMatrix>(layout)
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
                let layout = projected_batch_matrix(
                    &bound,
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    i,
                );
                g.matrix_transparent::<I, WP, W, ProjectedBatchMatrix>(layout)
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
                let layout = projected_grouped_matrix(
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    vector_size,
                    rows,
                    cols,
                );
                g.matrix_transparent::<I, WP, W, ProjectedGroupedMatrix>(layout)
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

    /// A plain matmul operand: one axis per edge, the split right down the middle.
    #[test]
    fn a_rank_two_operand_splits_between_its_axes() {
        let s = flat_space(&[(OH, 8), (CI, 4)]);
        assert_eq!(matrix_split(&s, 8, 4, 1), 1);
    }

    /// The convolution shape: the trailing *two* axes are the contraction, so `k` is their
    /// product and the split leaves only the output axis in the row group.
    #[test]
    fn a_contraction_over_two_axes_splits_before_both() {
        let s = flat_space(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(matrix_split(&s, 8, 8, 1), 1);
    }

    /// Both edges spanning several axes, which is what a 2-D convolution's input needs.
    #[test]
    fn both_groups_may_span_several_axes() {
        let s = flat_space(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(matrix_split(&s, 12, 2, 2), 2);
    }

    /// The smallest column group that reaches `cols` wins, so a degenerate leading axis stays in
    /// the row group rather than being swept into the column one. Either split addresses the same
    /// cells; taking the smaller keeps the answer deterministic.
    #[test]
    fn a_degenerate_leading_axis_stays_in_the_row_group() {
        let s = flat_space(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(matrix_split(&s, 1, 8, 1), 1);
    }

    /// No split gives the asked-for edges: the extents multiply to 32, and 8x8 is not a face.
    #[test]
    #[should_panic(expected = "no split of this tile's axes")]
    fn a_mismatched_fragment_is_refused() {
        let s = flat_space(&[(OH, 8), (RH, 2), (CI, 2)]);
        matrix_split(&s, 8, 8, 1);
    }

    /// The column group must contain the innermost axis: the view serves lines along `col`, so a
    /// split that leaves the vectorized axis in the row group has no line to read.
    #[test]
    #[should_panic(expected = "innermost (vectorized) axis")]
    fn the_vectorized_axis_must_land_in_the_column_group() {
        let s = flat_space(&[(OH, 8), (CI, 4)]);
        matrix_split(&s, 32, 1, 1);
    }

    /// The column edge and the innermost extent are both counted in lines, so a width that does
    /// not divide the innermost extent would scale the two by different amounts.
    #[test]
    #[should_panic(expected = "whole number of 4-wide lines")]
    fn a_partial_innermost_line_is_refused() {
        let s = flat_space(&[(OH, 8), (CI, 6)]);
        matrix_split(&s, 8, 6, 4);
    }
}
