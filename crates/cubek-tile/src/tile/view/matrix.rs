//! The 2-D matrix view over a [`Tile`]. [`BatchMatrix`] is a [`Layout`] that re-views the tile's
//! N-D [`Space`] as a plain [`Coords2d`] `(row, col)` matrix — leading batch axes pinned, trailing
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

#[cube]
impl<T: Numeric> Tile<T> {
    /// The leading axes are pinned to `i` unraveled over their extents. Only the
    /// (width-invariant) leading shape is read, off the window extent.
    fn batch_matrix(&self, i: usize) -> BatchMatrix {
        let rank = comptime!(self.space.rank());
        let shape = match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.extent(),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::matrix: a cmma fragment has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
        };
        let rows = comptime!(self.space.extent_at(rank - 2));
        // `cols` is a line count, so divide the innermost extent by the width.
        let w = self.vector_size();
        let cols = comptime!(self.space.extent_at(rank - 1) / w);

        let mut batches = CoordsDyn::new();

        #[unroll]
        for p in 0..rank - 2 {
            let weight = shape.fproduct(comptime!(((p + 1)..(rank - 2)).collect::<Vec<_>>()));
            batches.push(i.fcast::<u32>().fdiv(weight).frem(shape.at(p)));
        }

        BatchMatrix::new(batches, rows, cols)
    }

    /// The `i`-th batch matrix over `Vector<T, W>` lines (`W` = [`width`](Tile::width)).
    pub fn matrix<W: Size>(&self, i: usize) -> MatrixView<'_, Vector<T, W>> {
        let layout = self.batch_matrix(i);
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.masked::<W>(layout),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::matrix: a cmma fragment has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
        }
    }

    pub fn matrix_mut<W: Size>(&mut self, i: usize) -> MatrixViewMut<'_, Vector<T, W>> {
        let layout = self.batch_matrix(i);
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.masked_mut::<W>(layout),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::matrix_mut: a cmma fragment has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::matrix_mut: a tma source has no element view"),
        }
    }

    /// The `i`-th batch matrix as a quantization-transparent view: a plain tile serves a bare
    /// `Direct` read, a quantized one dequantizes each `(row, col)` per its scheme (`I`/`WP` the
    /// storage element and physical line, as [`copy_from`](Tile::copy_from) recovers them). The
    /// dequant-at-read twin of [`matrix`](Tile::matrix) — a leaf reads a quantized operand with no
    /// dequantize-into-`f32` fill.
    pub fn matrix_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        i: usize,
    ) -> TileView<'_, T, I, WP, W, Coords2d> {
        let layout = self.batch_matrix(i);
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.matrix_transparent::<I, WP, W>(layout),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::matrix_transparent: a cmma fragment has no memory view")
            }
            TileKind::TmaGmem(_) => {
                panic!("Tile::matrix_transparent: a tma source has no element view")
            }
        }
    }
}
