pub mod flat;
pub mod masked;
pub mod matrix;
pub mod quant;

pub use flat::*;
pub use masked::*;
pub use matrix::*;
pub use quant::*;

use cubecl::{prelude::*, std::tensor::layout::Coordinates};

/// A quantization-transparent read view over a [`Tile`](crate::Tile) serving `O`: `Direct` reads
/// the buffer as-is; `Quantized` reads the storage element `I` and dequantizes per the store's
/// [`QuantInfo`](crate::QuantInfo). Which variant a tile yields is comptime, so the plain path
/// compiles to a bare masked read.
#[derive(CubeType)]
pub enum TileView<'a, O: Numeric, I: Numeric, W: Size, C: Coordinates + 'a> {
    Direct(MaskedView<'a, Vector<O, W>, C>),
    Quantized(QuantizedView<'a, O, I, W, C>),
}

#[cube]
impl<'a, O: Numeric, I: Numeric, W: Size, C: Coordinates + 'a> TileView<'a, O, I, W, C> {
    pub fn read(&self, pos: C) -> Vector<O, W> {
        match self {
            TileView::Direct(direct) => direct.read(pos),
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
