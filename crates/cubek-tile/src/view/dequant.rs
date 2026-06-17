use std::marker::PhantomData;

use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::Coordinates,
};

use crate::MaskedView;

#[derive(CubeType)]
pub struct DequantView<'a, T: CubePrimitive, C: Coordinates, S: Numeric, F: Numeric> {
    pub values: MaskedView<'a, T, C>,
    pub scales: S,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _phantom: PhantomData<F>,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates, S: Numeric, F: Numeric> DequantView<'a, T, C, S, F> {
    pub fn new(values: MaskedView<'a, T, C>, scales: S, #[comptime] scheme: QuantScheme) -> Self {
        DequantView::<'a, T, C, S, F> {
            values,
            scales,
            scheme,
            _phantom: PhantomData,
        }
    }

    pub fn read(&self, pos: C) -> F {
        match comptime!(self.scheme.store) {
            QuantStore::Native => F::cast_from(self.values.read(pos)) * F::cast_from(self.scales),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}
