use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::Coordinates,
};

use crate::*;

/// The `Quantized` arm of a [`TileView`]: a masked view over the storage element `I` the buffer
/// truly holds, plus the scale/scheme to fold each read back into the served `O`.
#[derive(CubeType)]
pub struct QuantizedView<'a, O: Numeric, I: Numeric, W: Size, C: Coordinates + 'a> {
    values: MaskedView<'a, Vector<I, W>, C>,
    /// Per-tensor scale, broadcast across the line on read.
    scale: O,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, O: Numeric, I: Numeric, W: Size, C: Coordinates + 'a> QuantizedView<'a, O, I, W, C> {
    pub fn new(
        values: MaskedView<'a, Vector<I, W>, C>,
        scale: O,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, O, I, W, C> {
            values,
            scale,
            scheme,
        }
    }

    pub fn read(&self, pos: C) -> Vector<O, W> {
        let raw = Vector::<O, W>::cast_from(self.values.read(pos));
        match comptime!(self.scheme.store) {
            QuantStore::Native => raw * Vector::cast_from(self.scale),
            _ => panic!("only native quantization storage is supported for now"),
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}
