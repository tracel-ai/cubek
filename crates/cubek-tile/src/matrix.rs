//! The 2-D matrix view over a [`Tile`](super::Tile). A tile carries an N-D
//! [`Space`](super::Space); [`MatrixWindow`] is a [`Layout`] re-viewing it as a plain
//! [`Coords2d`] matrix by pinning the leading batch axes and exposing the trailing two.

use super::*;
use cubecl::{
    prelude::*,
    std::tensor::{
        View, ViewMut,
        layout::{Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

/// A [`Layout`] mapping a matrix coordinate `(row, col)` to the tile's source
/// coordinate `[batches…, row, col]`: leading batch axes pinned, trailing two exposed.
#[derive(CubeType, Clone)]
pub struct MatrixWindow {
    batches: CoordsDyn,
    tile_shape: Coords2d,
}

#[cube]
impl MatrixWindow {
    pub fn new(batches: CoordsDyn, #[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        MatrixWindow {
            batches,
            tile_shape: (
                u32::from_int(comptime!(rows as i64)),
                u32::from_int(comptime!(cols as i64)),
            ),
        }
    }
}

#[cube]
impl Layout for MatrixWindow {
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
        let mut count = u32::from_int(1);
        #[unroll]
        for p in 0..comptime!(self.space.rank() - 2) {
            count *= shape[p];
        }
        count as usize
    }

    /// The leading axes are pinned to `i` unraveled over their extents.
    fn matrix_window(&self, i: usize) -> MatrixWindow {
        let rank = comptime!(self.space.rank());
        let shape = self.view().shape();
        let rows = comptime!(self.space.extent(self.space.axis_at(rank - 2)));
        let cols = comptime!(self.space.extent(self.space.axis_at(rank - 1)));

        // Unravel `i` (row-major) over the leading extents into pinned batch coords.
        let mut batches = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(rank - 2) {
            let mut weight = u32::from_int(1);
            #[unroll]
            for q in comptime!(p + 1)..comptime!(rank - 2) {
                weight *= shape[q];
            }
            batches.push((i as u32 / weight) % shape[p]);
        }
        MatrixWindow::new(batches, rows, cols)
    }

    pub fn matrix(&self, i: usize) -> View<'_, T, Coords2d> {
        let layout = self.matrix_window(i);
        self.view().view(layout)
    }

    pub fn matrix_mut(&mut self, i: usize) -> ViewMut<'_, T, Coords2d> {
        let layout = self.matrix_window(i);
        self.view_mut().view_mut(layout)
    }
}

#[cube]
pub fn copy_2d<T: CubePrimitive>(dst: &mut ViewMut<'_, T, Coords2d>, src: &View<'_, T, Coords2d>) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}
