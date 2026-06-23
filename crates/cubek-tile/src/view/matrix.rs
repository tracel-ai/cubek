//! The 2-D matrix view over a [`Tile`]. [`BatchMatrix`] is a [`Layout`] that re-views the tile's
//! N-D [`Space`] as a plain [`Coords2d`] `(row, col)` matrix — leading batch axes pinned, trailing
//! two exposed; [`Tile::matrix`]/[`Tile::matrix_mut`] then wrap it as a [`MatrixView`]/
//! [`MatrixViewMut`] (a [`MaskedView`] carrying the comptime overhang-`check` flag). Used by the
//! matmul leaves and [`copy_2d()`].

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 2-D ([`BatchMatrix`]) view: one batch matrix of a [`Tile`].
pub type MatrixView<'a, T> = MaskedView<'a, T, Coords2d>;
/// The mutable twin of [`MatrixView`].
pub type MatrixViewMut<'a, T> = MaskedViewMut<'a, T, Coords2d>;

/// A [`Layout`] mapping `(row, col)` to a 1-D buffer offset by `base + row·row_stride +
/// col·col_stride`. For a contiguous-block leaf the base and strides are derived once, so the
/// hot loop reads with two multiply-adds instead of re-splitting the `[grid, leaf]` tiling
/// (a divmod) per element.
#[derive(CubeType, Clone)]
pub struct AffineLayout {
    base: u32,
    row_stride: u32,
    col_stride: u32,
    tile_shape: Coords2d,
}

#[cube]
impl AffineLayout {
    pub fn new(base: u32, row_stride: u32, col_stride: u32, shape: Coords2d) -> Self {
        AffineLayout {
            base,
            row_stride,
            col_stride,
            tile_shape: shape,
        }
    }
}

#[cube]
impl Layout for AffineLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        (self.base + row * self.row_stride + col * self.col_stride) as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
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

/// A [`Layout`] mapping a matrix coordinate `(row, col)` to the tile's source
/// coordinate `[batches…, row, col]`: leading batch axes pinned, trailing two exposed.
#[derive(CubeType, Clone)]
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
impl<T: CubePrimitive> Tile<T> {
    /// The product of the leading (batch) extents.
    pub fn matrix_count(&self) -> usize {
        let shape = self.view().shape();
        let mut count = 1;

        #[unroll]
        for p in 0..comptime!(self.space.rank() - 2) {
            count *= shape[p];
        }

        count as usize
    }

    /// The leading axes are pinned to `i` unraveled over their extents.
    fn batch_matrix(&self, i: usize) -> BatchMatrix {
        let rank = comptime!(self.space.rank());
        let shape = self.view().shape();
        let rows = comptime!(self.space.extent_at(rank - 2));
        let cols = comptime!(self.space.extent_at(rank - 1));

        let mut batches = CoordsDyn::new();

        #[unroll]
        for p in 0..rank - 2 {
            let mut weight = 1;

            #[unroll]
            for q in comptime!(p + 1)..rank - 2 {
                weight *= shape[q];
            }
            batches.push((i as u32 / weight) % shape[p]);
        }

        BatchMatrix::new(batches, rows, cols)
    }

    pub fn matrix(&self, i: usize) -> MatrixView<'_, T> {
        let layout = self.batch_matrix(i);
        match &self.payload {
            Payload::Gmem(g) => g.masked(layout),
            Payload::Smem(g) => g.masked(layout),
            Payload::Cmma(_) => panic!("Tile::matrix: a cmma fragment has no memory view"),
            Payload::TmaGmem(_) => panic!("Tile::matrix: a tma source has no element view"),
        }
    }

    pub fn matrix_mut(&mut self, i: usize) -> MatrixViewMut<'_, T> {
        let layout = self.batch_matrix(i);
        match &mut self.payload {
            Payload::Gmem(g) => g.masked_mut(layout),
            Payload::Smem(g) => g.masked_mut(layout),
            Payload::Cmma(_) => panic!("Tile::matrix_mut: a cmma fragment has no memory view"),
            Payload::TmaGmem(_) => panic!("Tile::matrix_mut: a tma source has no element view"),
        }
    }
}

#[cube]
pub fn copy_2d<T: CubePrimitive>(dst: &mut MatrixViewMut<'_, T>, src: &MatrixView<'_, T>) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            // `src` zeroes reads past its logical bound (the partial-tile overhang); the
            // staged buffer is unchecked, so the full padded cell is still written.
            dst.write((i, j), src.read((i, j)));
        }
    }
}
