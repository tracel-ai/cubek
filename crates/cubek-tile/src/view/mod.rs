pub mod flat;
pub mod matrix;

pub use flat::*;
pub use matrix::*;

use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coordinates},
};

/// A masked view over a [`Tile`](crate::Tile): a [`View`] re-shaped by some
/// [`Layout`](cubecl::std::tensor::layout::Layout) (a 2-D [`BatchMatrix`] or a 1-D [`FlatLayout`])
/// plus its own comptime bounds-check flag, so the
/// leaf reads it without being asked. `check` zeroes reads / skips writes past the operand's
/// logical bound (the partial-tile overhang); `false` is the unchecked fast path. The coordinate
/// type `C` is whatever the layout exposes — `Coords2d` for a matrix, `Coords1d` for a flat scan.
#[derive(CubeType)]
pub struct MaskedView<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: View<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> MaskedView<'a, T, C> {
    pub fn new(view: View<'a, T, C>, #[comptime] check: bool) -> Self {
        MaskedView::<'a, T, C> { view, check }
    }

    pub fn read(&self, pos: C) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            self.view.read(pos)
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}

/// The mutable twin of [`MaskedView`]. Its `write` skips the overhang under `check`, matching
/// the masked reads.
#[derive(CubeType)]
pub struct MaskedViewMut<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: ViewMut<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> MaskedViewMut<'a, T, C> {
    pub fn new(view: ViewMut<'a, T, C>, #[comptime] check: bool) -> Self {
        MaskedViewMut::<'a, T, C> { view, check }
    }

    pub fn read(&self, pos: C) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            self.view.read(pos)
        }
    }

    pub fn write(&mut self, pos: C, value: T) {
        if comptime!(self.check) {
            self.view.write_checked(pos, value);
        } else {
            self.view.write(pos, value);
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}
