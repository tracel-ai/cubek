use super::{
    filter::{SeparableFilter, SeparableFilterFamily, SeparableWeights, TapDistance},
    space::*,
};
use cubecl::{ir::ElemType, prelude::*};
use cubek_tile::*;

#[cube]
fn tap_distance<E: Float>(
    #[comptime] tap: Axis,
    #[comptime] output: Axis,
    scale: u32,
    offset: i32,
    divisor: u32,
    #[comptime] radius: usize,
) -> TapDistance<E> {
    sum_of(
        affine_along(tap, E::new(-(radius as f32)), E::new(1.0_f32)),
        Phase::<E> {
            coefficient: E::new(-1.0_f32),
            numerator_scale: scale,
            numerator_offset: offset,
            divisor,
            axis: output,
        },
    )
}

#[cube(launch)]
pub fn interpolate_tile_kernel<E: Float, V: Size, F: SeparableFilterFamily>(
    input: &TileArg<'_, E, V>,
    output: &TileArg<'_, E, V>,
    row_scale: u32,
    row_offset: i32,
    row_divisor: u32,
    col_scale: u32,
    col_offset: i32,
    col_divisor: u32,
    #[comptime] radius: usize,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));

    let row = tap_distance(TAP_H, OUTPUT_H, row_scale, row_offset, row_divisor, radius);
    let col = tap_distance(TAP_W, OUTPUT_W, col_scale, col_offset, col_divisor, radius);
    let weights = Tile::<E>::procedural::<SeparableWeights<E, F::Filter<E>>>(
        comptime!(space.project(&[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W])),
        product_of(F::Filter::<E>::along(row), F::Filter::<E>::along(col)),
        comptime!(output.spec.leaf),
    );

    let mut output = output.tile(space);
    output.zero();
    output.mma(&weights, &input);
}
