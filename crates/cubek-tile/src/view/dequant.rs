use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::Coordinates,
};

use crate::*;

/// A dequantizing view over a [`Tile`]: wraps the tile's raw masked view and folds the scale back in
/// on every [`read`](DequantView::read), yielding the float compute type `F` (wider than the
/// quantized storage `I`). Paired with the plain branch inside [`TileView`] so an elementwise leaf
/// reads a quantized tile exactly like a plain one.
#[derive(CubeType)]
pub struct DequantView<'a, I: Numeric, N: Size, F: Numeric, C: Coordinates + 'a> {
    values: MaskedView<'a, Vector<I, N>, C>,
    /// Per-tensor scale, broadcast across the line on read.
    scale: F,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, I: Numeric, N: Size, F: Numeric, C: Coordinates + 'a> DequantView<'a, I, N, F, C> {
    pub fn new(
        values: MaskedView<'a, Vector<I, N>, C>,
        scale: F,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        DequantView::<'a, I, N, F, C> {
            values,
            scale,
            scheme,
        }
    }

    // very naive impl that only supports native quantization storage
    pub fn read(&self, pos: C) -> Vector<F, N> {
        let raw = Vector::<F, N>::cast_from(self.values.read(pos));
        match comptime!(self.scheme.store) {
            QuantStore::Native => raw * Vector::<F, N>::cast_from(self.scale),
            _ => unimplemented!("only native quantization storage is supported"),
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}
