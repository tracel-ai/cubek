use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::Coordinates,
};

use crate::*;

#[derive(CubeType)]
pub struct QuantizedView<'a, I: Numeric, N: Size, F: Numeric, C: Coordinates + 'a> {
    values: MaskedView<'a, Vector<I, N>, C>,
    /// Per-tensor scale, broadcast across the line on read.
    scale: F,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, I: Numeric, N: Size, F: Numeric, C: Coordinates + 'a> QuantizedView<'a, I, N, F, C> {
    pub fn new(
        values: MaskedView<'a, Vector<I, N>, C>,
        scale: F,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, I, N, F, C> {
            values,
            scale,
            scheme,
        }
    }

    // very naive impl that only supports native quantization storage
    pub fn read(&self, pos: C) -> Vector<F, N> {
        let raw = Vector::<F, N>::cast_from(self.values.read(pos));
        let scale = Vector::<F, N>::cast_from(self.scale);
        match comptime!(self.scheme.store) {
            QuantStore::Native => raw * scale,
            _ => panic!("only native quantization storage is supported for now"),
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}
