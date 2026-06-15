//! The 2-D matrix view over a [`Tile`](super::Tile). A tile carries an N-D
//! [`Space`](super::Space); [`BatchMatrix`] is a [`Layout`] re-viewing it as a plain
//! [`Coords2d`] matrix by pinning the leading batch axes and exposing the trailing two.
//! [`MaskedView`] wraps the resulting view with a comptime `check` flag.

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

    pub fn matrix(&self, i: usize) -> MaskedView<'_, T> {
        let layout = self.batch_matrix(i);
        match &self.payload {
            Payload::Gmem(g) | Payload::Smem(g) => g.masked(layout),
            Payload::Cmma(_) => panic!("Tile::matrix: a cmma fragment has no memory view"),
        }
    }

    pub fn matrix_mut(&mut self, i: usize) -> MaskedViewMut<'_, T> {
        let layout = self.batch_matrix(i);
        match &mut self.payload {
            Payload::Gmem(g) | Payload::Smem(g) => g.masked_mut(layout),
            Payload::Cmma(_) => panic!("Tile::matrix_mut: a cmma fragment has no memory view"),
        }
    }
}

/// A [`Tile`]'s `i`-th batch matrix as a 2-D view that carries its own comptime
/// bounds-check flag, so the leaf reads it without being asked. `check` zeroes reads /
/// skips writes past the operand's logical bound (the partial-tile overhang); `false`
/// is the unchecked fast path.
#[derive(CubeType)]
pub struct MaskedView<'a, T: CubePrimitive> {
    view: View<'a, T, Coords2d>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive> MaskedView<'a, T> {
    pub fn new(view: View<'a, T, Coords2d>, #[comptime] check: bool) -> Self {
        MaskedView::<'a, T> { view, check }
    }

    pub fn read(&self, pos: Coords2d) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            self.view.read(pos)
        }
    }

    pub fn shape(&self) -> Coords2d {
        self.view.shape()
    }
}

/// The mutable twin of [`MaskedView`]. Its `write` skips the overhang under `check`, matching
/// the masked reads.
#[derive(CubeType)]
pub struct MaskedViewMut<'a, T: CubePrimitive> {
    view: ViewMut<'a, T, Coords2d>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive> MaskedViewMut<'a, T> {
    pub fn new(view: ViewMut<'a, T, Coords2d>, #[comptime] check: bool) -> Self {
        MaskedViewMut::<'a, T> { view, check }
    }

    pub fn read(&self, pos: Coords2d) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            self.view.read(pos)
        }
    }

    pub fn write(&mut self, pos: Coords2d, value: T) {
        if comptime!(self.check) {
            self.view.write_checked(pos, value);
        } else {
            self.view.write(pos, value);
        }
    }

    pub fn shape(&self) -> Coords2d {
        self.view.shape()
    }
}

#[cube]
pub fn copy_2d<T: CubePrimitive>(dst: &mut MaskedViewMut<'_, T>, src: &MaskedView<'_, T>) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            // `src` zeroes reads past its logical bound (the partial-tile overhang); the
            // staged buffer is unchecked, so the full padded cell is still written.
            dst.write((i, j), src.read((i, j)));
        }
    }
}
