use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::{Coordinates, Coords1d},
};

use crate::*;

/// A dequantizing flat view over a [`Tile`]: wraps the tile's raw [`FlatView`] and folds the scale
/// back in on every [`read`](DequantView::read), returning the float compute type `F`. When the tile
/// is not quantized (`scheme = None`) it is a pure passthrough cast, so the *same* leaf reads a plain
/// or a quantized tile without branching. Built by [`Tile::flat_dequant`].
#[derive(CubeType)]
pub struct DequantView<'a, T: CubePrimitive, F: Numeric, C: Coordinates + 'a, N: Size> {
    values: MaskedView<'a, Vector<T, N>, C>,
    scale: F,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, T: CubePrimitive, F: Numeric, C: Coordinates + 'a, N: Size> DequantView<'a, T, F, C, N> {
    pub fn new(
        values: MaskedView<'a, Vector<T, N>, C>,
        scale: F,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        DequantView::<'a, T, F, C> {
            values,
            scale,
            scheme,
        }
    }

    pub fn read(&self, pos: C) -> T {
        let raw = self.values.read(pos);
        match comptime!(self.scheme.store) {
            QuantStore::Native => Vector::cast_from(raw) * T::cast_from(self.scale),
            _ => unimplemented!("only native quantization storage is supported"),
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}
