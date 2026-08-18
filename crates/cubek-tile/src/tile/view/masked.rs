use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coordinates},
};

/// A masked view over a [`Tile`](crate::Tile): a [`View`] re-shaped by some layout plus
/// its own comptime `check` flag, so the leaf zeroes reads / skips writes past the
/// partial-tile overhang; `false` is the unchecked fast path.
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
            // `check == false` means the launch proved this access in-bounds; dropping
            // the inner view's redundant index clamp speeds up the hot leaf loop.
            self.view.read_unchecked(pos)
        }
    }

    /// Whether `pos` lands on the operand's real data (`true` unconditionally when `check` is
    /// `false`: the launch already proved every access in-bounds). A caller that cannot use
    /// [`read`](Self::read)'s zeroed default, because its fold's identity is not zero (`Max`,
    /// `Min`), queries this and selects its own fallback instead.
    pub fn is_in_bounds(&self, pos: C) -> bool {
        if comptime!(self.check) {
            self.view.is_in_bounds(pos)
        } else {
            true.runtime()
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
            self.view.read_unchecked(pos)
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
