pub mod dequant;
pub mod flat;
pub mod masked;
pub mod matrix;

pub use dequant::*;
pub use flat::*;
pub use masked::*;
pub use matrix::*;

use cubecl::{prelude::*, std::tensor::layout::Coordinates};

/// A quantization-transparent view over a [`Tile`]: either a plain [`MaskedView`] or a
/// [`DequantView`]. Both branches `read` into the same compute type `Vector<F, N>` (`F` defaults to
/// `f32`), so an elementwise leaf treats a quantized and a plain tile identically — the dequant
/// happens underneath. Which branch is taken is fixed at comptime when [`Tile::flat`] builds it.
#[derive(CubeType)]
pub enum TileView<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric = f32> {
    Masked(MaskedView<'a, Vector<I, N>, C>),
    Dequant(DequantView<'a, I, N, F, C>),
}

#[cube]
impl<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric> TileView<'a, I, N, C, F> {
    pub fn read(&self, pos: C) -> Vector<F, N> {
        match self {
            TileView::Masked(masked) => Vector::cast_from(masked.read(pos)),
            TileView::Dequant(dequant) => dequant.read(pos),
        }
    }

    pub fn shape(&self) -> C {
        match self {
            TileView::Masked(masked) => masked.shape(),
            TileView::Dequant(dequant) => dequant.shape(),
        }
    }
}

#[derive(CubeType)]
pub enum TileViewMut<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric = f32> {
    Masked(MaskedViewMut<'a, Vector<I, N>, C>),
    Dequant(DequantView<'a, I, N, F, C>),
}

#[cube]
impl<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric> TileViewMut<'a, I, N, C, F> {
    pub fn read(&self, pos: C) -> Vector<F, N> {
        match self {
            TileViewMut::Masked(masked) => Vector::cast_from(masked.read(pos)),
            TileViewMut::Dequant(dequant) => dequant.read(pos),
        }
    }

    pub fn shape(&self) -> C {
        match self {
            TileViewMut::Masked(masked) => masked.shape(),
            TileViewMut::Dequant(dequant) => dequant.shape(),
        }
    }

    pub fn write(&mut self, pos: C, value: Vector<I, N>) {
        match self {
            TileViewMut::Masked(masked) => masked.write(pos, value),
            TileViewMut::Dequant(_) => unimplemented!(),
        }
    }
}
