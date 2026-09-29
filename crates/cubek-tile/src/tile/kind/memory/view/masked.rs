use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coordinates},
};

/// A [`View`] plus a comptime `check` flag: reads past the overhang zero, writes skip.
#[derive(CubeType)]
pub struct Masked<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: View<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> Masked<'a, T, C> {
    pub fn new(view: View<'a, T, C>, #[comptime] check: bool) -> Self {
        Masked::<'a, T, C> { view, check }
    }

    pub fn read(&self, pos: C) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            // `check == false`: the launch proved this access in-bounds.
            self.view.read_unchecked(pos)
        }
    }

    /// Whether `pos` lands on real data; always `true` when `check` is `false`.
    pub fn is_in_bounds(&self, pos: C) -> bool {
        if comptime!(self.check) {
            self.view.is_in_bounds(pos)
        } else {
            true.runtime()
        }
    }

    /// Whether the non-empty box at `pos` with `extent` is wholly in bounds.
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        if comptime!(self.check) {
            let one = C::from_int(pos.clone(), 1i64);
            let far = C::sub(C::add(pos, extent), one);
            self.view.is_in_bounds(far)
        } else {
            true.runtime()
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}

/// The mutable twin of [`Masked`].
#[derive(CubeType)]
pub struct MaskedMut<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: ViewMut<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> MaskedMut<'a, T, C> {
    pub fn new(view: ViewMut<'a, T, C>, #[comptime] check: bool) -> Self {
        MaskedMut::<'a, T, C> { view, check }
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

    /// Mutable counterpart to [`Masked::block_in_bounds`].
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        if comptime!(self.check) {
            let one = C::from_int(pos.clone(), 1i64);
            let far = C::sub(C::add(pos, extent), one);
            self.view.is_in_bounds(far)
        } else {
            true.runtime()
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}
