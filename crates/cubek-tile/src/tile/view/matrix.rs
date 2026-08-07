//! The 2-D matrix view over a [`Tile`]. [`BatchMatrix`] is a [`Layout`] that re-views the tile's
//! N-D [`Space`] as a plain [`Coords2d`] `(row, col)` matrix: leading batch axes pinned, trailing
//! two exposed; [`Tile::matrix`]/[`Tile::matrix_mut`] then wrap it as a [`MatrixView`]/
//! [`MatrixViewMut`] (a [`MaskedView`] carrying the comptime overhang-`check` flag). Used by the
//! matmul leaves and [`copy_2d()`].

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
    batches: CoordsDyn,
    tile_shape: Coords2d,
}

#[cube]
impl BatchMatrix {
    pub fn new(batches: CoordsDyn, #[comptime] rows: usize, #[comptime] cols: usize) -> Self {
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
        let mut out = self.batches.clone();
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

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct ProjectedBatchMatrix {
    batches: CoordsDyn,
    tile_shape: Coords2d,
    #[cube(comptime)]
    projection: Projection,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl ProjectedBatchMatrix {
    pub fn new(
        batches: CoordsDyn,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
        #[comptime] projection: Projection,
        #[comptime] space: Space,
    ) -> Self {
        ProjectedBatchMatrix {
            batches,
            tile_shape: (rows as u32, cols as u32).runtime(),
            projection,
            space,
        }
    }
}

#[cube]
impl Layout for ProjectedBatchMatrix {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        let mut logical = self.batches.clone();
        logical.push(t0);
        logical.push(t1);

        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            let n = comptime!(self.projection.physical_axis(pa).terms().len());
            let mut terms = Coords::<u32>::new();
            #[unroll]
            for t in 0..n {
                let term = comptime!(self.projection.physical_axis(pa).terms()[t]);
                let p = comptime!(self.space.position(term.axis));
                terms.push(logical[p].fmul(comptime!(term.scale.get() as u32)));
            }
            out.push(terms.fsum(comptime!((0..n).collect::<Vec<_>>())));
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
        let (t0, t1) = pos;
        let (s0, s1) = self.tile_shape;
        t0 < s0 && t1 < s1
    }
}

#[cube]
fn batch_matrix(
    bound: Coords<u32>,
    #[comptime] space: Space,
    #[comptime] vector_size: usize,
    i: usize,
) -> BatchMatrix {
    let rank = comptime!(space.rank());
    let shape = bound;
    let rows = comptime!(space.extent_at(rank - 2));
    let cols = comptime!(space.extent_at(rank - 1) / vector_size);

    let mut batches = CoordsDyn::new();

    #[unroll]
    for p in 0..rank - 2 {
        let weight = shape.fproduct(comptime!(((p + 1)..(rank - 2)).collect::<Vec<_>>()));
        batches.push(i.fcast::<u32>().fdiv(weight).frem(shape.at(p)));
    }

    BatchMatrix::new(batches, rows, cols)
}

#[cube]
fn projected_batch_matrix(
    bound: Coords<u32>,
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    #[comptime] vector_size: usize,
    i: usize,
) -> ProjectedBatchMatrix {
    let rank = comptime!(space.rank());
    let rows = comptime!(space.extent_at(rank - 2));
    let cols = comptime!(space.extent_at(rank - 1) / vector_size);

    let mut batches = CoordsDyn::new();

    #[unroll]
    for p in 0..rank - 2 {
        let mut weight = 1u32.runtime();
        #[unroll]
        for q in p + 1..rank - 2 {
            let axis_q = comptime!(space.axis_at(q));
            let p_idx = comptime!(space.position(axis_q));
            let raw_q = bound.at(p_idx).fcast::<u32>();
            let w_q = comptime!(if p_idx == rank - 1 { vector_size } else { 1 });
            weight *= comptime!(w_q as u32).runtime() * raw_q;
        }
        let axis_p = comptime!(space.axis_at(p));
        let p_idx = comptime!(space.position(axis_p));
        let raw_p = bound.at(p_idx).fcast::<u32>();
        let w_p = comptime!(if p_idx == rank - 1 { vector_size } else { 1 });
        let extent = comptime!(w_p as u32).runtime() * raw_p;

        batches.push(i.fcast::<u32>().fdiv(weight).frem(extent));
    }

    ProjectedBatchMatrix::new(batches, rows, cols, projection, comptime!(space))
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// The `i`-th batch matrix over `Vector<T, W>` lines (`W` = [`width`](Tile::width)).
    pub fn matrix<W: Size>(&self, i: usize) -> MatrixView<'_, Vector<T, W>> {
        let vector_size = self.vector_size();
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.projection.is_direct()) {
                    let layout =
                        batch_matrix(g.extent(), comptime!(self.space.clone()), vector_size, i);
                    g.masked::<W, Coords2d, BatchMatrix>(layout)
                } else {
                    let layout = projected_batch_matrix(
                        g.extent(),
                        comptime!(self.space.clone()),
                        comptime!(g.projection.clone()),
                        vector_size,
                        i,
                    );
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
                if comptime!(g.projection.is_direct()) {
                    let layout =
                        batch_matrix(g.extent(), comptime!(self.space.clone()), vector_size, i);
                    g.masked_mut::<W>(layout)
                } else {
                    let layout = projected_batch_matrix(
                        g.extent(),
                        comptime!(self.space.clone()),
                        comptime!(g.projection.clone()),
                        vector_size,
                        i,
                    );
                    g.masked_mut_projected::<W>(layout)
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
                if comptime!(g.projection.is_direct()) {
                    let layout =
                        batch_matrix(g.extent(), comptime!(self.space.clone()), vector_size, i);
                    g.matrix_transparent::<I, WP, W>(layout)
                } else {
                    let layout = projected_batch_matrix(
                        g.extent(),
                        comptime!(self.space.clone()),
                        comptime!(g.projection.clone()),
                        vector_size,
                        i,
                    );
                    g.matrix_transparent_projected::<I, WP, W>(layout)
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
}
