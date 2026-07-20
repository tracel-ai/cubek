pub mod accumulate;
pub mod flat;
pub mod masked;
pub mod matrix;
pub mod quant;

pub use accumulate::*;
pub use flat::*;
pub use masked::*;
pub use matrix::*;
pub use quant::*;

use cubecl::{prelude::*, std::tensor::layout::Coordinates};

/// A quantization-transparent read view over a [`Tile`](crate::Tile) serving `O`: `Direct`
/// reads the buffer as-is; `Quantized` reads the storage element `I` and dequantizes per the
/// store's [`QuantInfo`](crate::QuantInfo). Which variant a tile yields is comptime, so the plain
/// path compiles to a bare masked read. The coordinate `C` is [`Coords1d`](cubecl::std::tensor::layout::Coords1d)
/// for a flat scan ([`flat_transparent`](crate::MemData::flat_transparent), the fill) or
/// [`Coords2d`](cubecl::std::tensor::layout::Coords2d) for the matrix leaf
/// ([`matrix_transparent`](crate::MemData::matrix_transparent), direct dequant serving).
///
/// `WP` is the physical line width, `W` the served one; they coincide except on a packed store
/// (see [`QuantizedView`]), and `Direct` (never quantized) always serves what it holds.
#[derive(CubeType)]
pub enum TileView<'a, O: Numeric, I: Numeric, WP: Size, W: Size, C: Coordinates + 'a> {
    Direct(MaskedView<'a, Vector<O, W>, C>),
    Quantized(QuantizedView<'a, I, WP, C>),
}

#[cube]
impl<'a, O: Numeric, I: Numeric, WP: Size, W: Size, C: Coordinates + 'a>
    TileView<'a, O, I, WP, W, C>
{
    pub fn read(&self, pos: C) -> Vector<O, W> {
        match self {
            TileView::Direct(direct) => direct.read(pos),
            TileView::Quantized(quant) => quant.read::<O, W>(pos),
        }
    }

    pub fn shape(&self) -> C {
        match self {
            TileView::Direct(direct) => direct.shape(),
            TileView::Quantized(quant) => quant.shape(),
        }
    }

    /// The comptime overhang-mask flag of the underlying view, so a leaf can make the same
    /// unroll decision it makes on a plain [`MatrixView`](crate::MatrixView). Both arms carry a
    /// [`MaskedView`], whose `check` is the flag.
    pub fn check(&self) -> comptime_type!(bool) {
        match self {
            TileView::Direct(direct) => comptime!(direct.check),
            TileView::Quantized(quant) => comptime!(quant.values.check),
        }
    }
}
