//! The 2-D matrix views over a [`Tile`]. [`TileMatrix`] is a [`Layout`] re-viewing the tile's N-D
//! [`Space`] as a [`Coords2d`] `(row, col)` matrix: a pinned batch prefix, then a row group and a
//! column group of as many axes as each edge needs. [`Tile::matrix`] wraps it as a [`MatrixView`].

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 2-D ([`TileMatrix`]) view: one matrix of a [`Tile`].
pub(crate) type MatrixView<'a, T> = Masked<'a, T, Coords2d>;
/// The mutable twin of [`MatrixView`].
pub(crate) type MatrixViewMut<'a, T> = MaskedMut<'a, T, Coords2d>;

/// A [`Layout`] presenting a tile's logical box as one `(row, col)` matrix over three axes of
/// axes, in the space's own order: a *batch* prefix already pinned to one matrix, then the axes
/// `row` unravels over, then the axes `col` unravels over (the innermost a line count).
///
/// One type because there is one concept. A plain batched matmul exposes exactly two axes, so the
/// unravels are the identity. A convolution contracts over taps *and* channels, so its `k` group
/// holds several; a [partitioned](Composition::Disjoint) `(M, KB, KI)` reads as `M·KB` by `KI`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct TileMatrix {
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
pub(crate) type ProjectedMatrix = Projected<TileMatrix>;

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
        let mut coords = self.batches.clone();
        coords.extend(&self.row_extents.unravel(row));
        coords.extend(&self.col_extents.unravel(col));
        coords.to_dyn()
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
        let (rows, cols) = self.tile_shape;
        row < rows && col < cols
    }
}

/// The leading (batch) extents a matrix index unravels over, in the space's axis order.
///
/// A direct operand reads them off the window, the only place a [`Dynamic`](crate::Extent::Dynamic)
/// top-level axis carries a size. A gathered one reads them off the space: its window is boxed in
/// *physical* axes, fewer than the logical ones and combinations of them, so none sizes a logical.
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
    // extent need not fill whole lines, and a checked read's box must include the partial last one
    // it really holds. `cols` is a shape here, never a stride, so this only widens the bound.
    let cols = comptime!(
        line_extents(space, vector_size, axes.col_split, rank)
            .iter()
            .product::<usize>()
    );
    let extents = leading_extents(bound, comptime!(space), gathered, comptime!(axes.row_split));

    TileMatrix::new(
        extents.unravel(i.cast::<u32>()),
        Coords::constant(comptime!(line_extents(
            space,
            vector_size,
            axes.row_split,
            axes.col_split
        ))),
        Coords::constant(comptime!(line_extents(
            space,
            vector_size,
            axes.col_split,
            rank
        ))),
        rows,
        cols,
    )
}

/// The logical coordinate of the value line at `(row, col)` of the `i`-th batch matrix a tile
/// reads as: one entry per axis of the tile's space, in scalars, the line's first value. What a
/// scale covering that line is looked up at ([`FactorReader`](crate::FactorReader)).
#[cube]
pub(crate) fn matrix_coords(
    row: u32,
    col: u32,
    i: usize,
    #[comptime] space: &Space,
    #[comptime] axes: MatrixAxes,
    #[comptime] vector_size: usize,
) -> Coords<u32> {
    let rank = comptime!(space.rank());
    let mut coords = Coords::<u32>::new();
    let batches = Coords::constant(comptime!(
        (0..axes.row_split)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ))
    .unravel(i.cast::<u32>());
    #[unroll]
    for p in 0..batches.len() {
        coords.push(batches.at(p));
    }
    let rows = Coords::constant(comptime!(
        (axes.row_split..axes.col_split)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ))
    .unravel(row);
    #[unroll]
    for p in 0..rows.len() {
        coords.push(rows.at(p));
    }
    // The column edge counts in lines, so its innermost digit is a line index; the value's own
    // coordinate is that many lines in.
    let cols = Coords::constant(comptime!(line_extents(
        space,
        vector_size,
        axes.col_split,
        rank
    )))
    .unravel(col);
    let n = cols.len();
    #[unroll]
    for p in 0..n {
        if comptime!(p == n - 1) {
            coords.push(cols.at(p).times(comptime!(vector_size as u32)));
        } else {
            coords.push(cols.at(p));
        }
    }
    coords
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
        Coords::constant(comptime!(line_extents(space, vector_size, 0, split))),
        Coords::constant(comptime!(line_extents(space, vector_size, split, rank))),
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
    /// The `i`-th batch matrix over the trailing two axes, in `Vector<T, W>` lines (`W` =
    /// [`vector_size`](Tile::vector_size)), read through whatever [`Packing`] this tile carries.
    pub fn matrix<W: Size>(&self, i: usize) -> MatrixView<'_, Vector<T, W>> {
        self.matrix_packed::<W>(comptime!(MatrixAxes::trailing(&self.place.space)), i)
    }

    /// The `i`-th batch matrix over the axes `axes` names, read through whatever [`Packing`]
    /// this tile carries: a plain tile as it stands, a packed one unpacked at the read, a
    /// quantized one dequantized per its scheme, with no dequantize-into-`f32` fill.
    pub fn matrix_packed<W: Size>(
        &self,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let g = self.mem("matrix");
        let layout = g.batch_matrix(comptime!(self.place.space.clone()), axes, i);
        g.packed::<W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
    }

    /// The fragment's grouped matrix, read through whatever [`Packing`] this tile carries. The
    /// manual-mma twin of [`matrix_packed`](Tile::matrix_packed).
    ///
    /// `cols` is scalar, as an mma definition states it; the view serves lines.
    pub(crate) fn fragment_matrix_packed<W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let g = self.mem("fragment_matrix");
        let layout = g.whole_matrix(comptime!(self.place.space.clone()), rows, cols);
        g.packed::<W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
    }

    /// [`fragment_matrix_packed`](Tile::fragment_matrix_packed) at a stated storage element `I`
    /// and physical line `WP`: several logical axes may flatten into one edge, so a contraction
    /// over taps *and* channels still has a `k` edge, read straight out of a compacted stage.
    pub fn fragment_matrix<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let g = self.mem("fragment_matrix");
        let layout = g.whole_matrix(comptime!(self.place.space.clone()), rows, cols);
        g.transparent::<I, WP, W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
    }
}
