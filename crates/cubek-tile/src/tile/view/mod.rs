pub mod flat;
pub mod masked;
pub mod matrix;
pub mod quant;

pub use flat::*;
pub use masked::*;
pub use matrix::*;
pub use quant::*;

use cubecl::{prelude::*, std::tensor::layout::Coords1d};

/// A quantization-transparent flat read view over a [`Tile`](crate::Tile) serving `O`: `Direct`
/// reads the buffer as-is; `Quantized` reads the storage element `I` and dequantizes per the
/// store's [`QuantInfo`](crate::QuantInfo). Which variant a tile yields is comptime, so the plain
/// path compiles to a bare masked read. Always 1-D ([`Coords1d`]) — its only producer is
/// [`flat`](crate::Tile::flat).
///
/// `WP` is the physical line width, `W` the served one; they coincide except on a packed store
/// (see [`QuantizedView`]), and `Direct` — never quantized — always serves what it holds.
#[derive(CubeType)]
pub enum TileView<'a, O: Numeric, I: Numeric, WP: Size, W: Size> {
    Direct(FlatView<'a, Vector<O, W>>),
    Quantized(QuantizedView<'a, I, WP>),
}

#[cube]
impl<'a, O: Numeric, I: Numeric, WP: Size, W: Size> TileView<'a, O, I, WP, W> {
    pub fn read(&self, pos: Coords1d) -> Vector<O, W> {
        match self {
            TileView::Direct(direct) => direct.read(pos),
            TileView::Quantized(quant) => quant.read::<O, W>(pos),
        }
    }

    pub fn shape(&self) -> Coords1d {
        match self {
            TileView::Direct(direct) => direct.shape(),
            TileView::Quantized(quant) => quant.shape(),
        }
    }
}
