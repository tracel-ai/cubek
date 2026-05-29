#![allow(non_snake_case)]

use cubecl::prelude::*;
use cubek_tile::*;

/// The Mosaic kernel: every operand is a [`Tile`]
/// The whole matmul is one line, driven by the accumulator.
#[cube(launch)]
pub fn mosaic_kernel<E: Numeric, S: Size>(
    a: &Tile<'_, E, S>,
    b: &Tile<'_, E, S>,
    c: &mut Tile<'_, E, S>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    c.mma(a, b);
}
