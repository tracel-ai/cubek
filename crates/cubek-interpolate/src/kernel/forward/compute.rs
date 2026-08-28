use super::{
    filter::{SeparableFilter, SeparableFilterFamily, SeparableWeights, TapDistance},
    space::*,
};
use cubecl::{ir::ElemType, prelude::*};
use cubek_tile::{
    Axis, Phase, Semiring, Space, Tile, TileArg, affine_along, separable_product, sum_of,
};

/// The distance from a tap to the source coordinate the output position lands on.
///
/// The rational is comptime because it is the same one the gather's projection already carries
/// ([`Rational::tap_axis`](super::coordinate::Rational::tap_axis)), and the kernel is already
/// compiled per shape. Handed as launch scalars instead, its divisor reaches [`Phase`] as a
/// runtime value: nothing folds, and every output position pays a true integer division for the
/// residue and a float division to normalize it, per resampled axis.
#[cube]
fn tap_distance<E: Float>(
    #[comptime] tap: Axis,
    #[comptime] output: Axis,
    #[comptime] scale: u32,
    #[comptime] offset: i32,
    #[comptime] divisor: u32,
    #[comptime] radius: usize,
) -> TapDistance<E> {
    sum_of(
        affine_along(tap, E::new(-(radius as f32)), E::new(1.0_f32)),
        Phase::<E> {
            coefficient: E::new(-1.0_f32),
            numerator_scale: comptime!(scale),
            numerator_offset: comptime!(offset),
            divisor: comptime!(divisor),
            axis: output,
        },
    )
}

#[cube(launch)]
pub fn interpolate_tile_kernel<E: Float, V: Size, F: SeparableFilterFamily>(
    input: &TileArg<'_, E, V>,
    output: &TileArg<'_, E, V>,
    #[comptime] row_scale: u32,
    #[comptime] row_offset: i32,
    #[comptime] row_divisor: u32,
    #[comptime] col_scale: u32,
    #[comptime] col_offset: i32,
    #[comptime] col_divisor: u32,
    #[comptime] radius: usize,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));

    let row = tap_distance(TAP_H, OUTPUT_H, row_scale, row_offset, row_divisor, radius);
    let col = tap_distance(TAP_W, OUTPUT_W, col_scale, col_offset, col_divisor, radius);
    let mut factors = Sequence::new();
    factors.push(F::Filter::<E>::along(row));
    factors.push(F::Filter::<E>::along(col));
    let weights = Tile::<E>::procedural_separable::<SeparableWeights<E, F::Filter<E>>>(
        comptime!(space.project(&[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W])),
        separable_product(factors),
    );
    let weights = match comptime!(F::NORMALIZATION) {
        Some((mask, guard)) => weights.normalized(comptime!(mask), comptime!(guard)),
        None => weights,
    };

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}
