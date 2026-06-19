pub mod flat;
pub mod masked;
pub mod matrix;
pub mod quant;

pub use flat::*;
pub use masked::*;
pub use matrix::*;
pub use quant::*;

use cubecl::{prelude::*, std::tensor::layout::Coordinates};

// A quantization-transparent view over a tile.
#[derive(CubeType)]
pub enum TileView<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric = f32> {
    Direct(MaskedView<'a, Vector<I, N>, C>),
    Quantized(QuantizedView<'a, I, N, F, C>),
}

#[cube]
impl<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric> TileView<'a, I, N, C, F> {
    pub fn read(&self, pos: C) -> Vector<F, N> {
        match self {
            TileView::Direct(direct) => Vector::cast_from(direct.read(pos)),
            TileView::Quantized(quant) => quant.read(pos),
        }
    }

    pub fn shape(&self) -> C {
        match self {
            TileView::Direct(direct) => direct.shape(),
            TileView::Quantized(quant) => quant.shape(),
        }
    }
}

#[derive(CubeType)]
pub enum TileViewMut<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric = f32> {
    Direct(MaskedViewMut<'a, Vector<I, N>, C>),
    Quantized(QuantizedView<'a, I, N, F, C>),
}

#[cube]
impl<'a, I: Numeric, N: Size, C: Coordinates + 'a, F: Numeric> TileViewMut<'a, I, N, C, F> {
    pub fn read(&self, pos: C) -> Vector<F, N> {
        match self {
            TileViewMut::Direct(direct) => Vector::cast_from(direct.read(pos)),
            TileViewMut::Quantized(quant) => quant.read(pos),
        }
    }

    pub fn shape(&self) -> C {
        match self {
            TileViewMut::Direct(direct) => direct.shape(),
            TileViewMut::Quantized(quant) => quant.shape(),
        }
    }

    pub fn write(&mut self, pos: C, value: Vector<I, N>) {
        match self {
            TileViewMut::Direct(direct) => direct.write(pos, value),
            TileViewMut::Quantized(_) => panic!("writing to quantized view is not supported yet"),
        }
    }
}
