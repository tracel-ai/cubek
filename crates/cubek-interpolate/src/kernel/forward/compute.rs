use super::{
    filter::{SeparableFilter, SeparableFilterFamily, SeparableWeights, TapDistance},
    space::*,
};
use crate::InputStage;
use cubecl::{ir::ElemType, prelude::*};
use cubek_tile::{
    Axis, Phase, RegisterBlock, Ring, Semiring, StageStorage, Tile, TileArg, affine_along,
    pipelined, separable_product, sum_of,
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
    #[comptime] plan: InterpolateSpace,
    #[comptime] stage: InputStage,
    #[comptime] padded: Option<usize>,
    #[comptime] config: RegisterBlock,
    #[define(E)] _dtype: ElemType,
) {
    let space = comptime!(plan.space());
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

    let output = output.tile(space);

    // This cube's box of the output, walked over the taps and its channel blocks. Whether the
    // input is staged into shared memory for each block is the launch's call on the window's
    // size, stated as `stage`; the walk is the same either way.
    let cubes = output
        .op_space(&weights, &input)
        .level(comptime!(plan.cube_level()));
    match comptime!(stage) {
        InputStage::Smem => {
            let mut ring =
                Ring::smem_single_at(&cubes, &input, StageStorage::Strided, padded, 1usize);
            pipelined(cubes, &mut ring, |slot, block| {
                let output_block = output.at(block);
                let weights_block = weights.at(block);
                slot.consume(|input_block| {
                    interpolate_block(&output_block, &weights_block, input_block, plan, config);
                });
            });
        }
        InputStage::InPlace => {
            for block in cubes {
                interpolate_block(
                    &output.at(&block),
                    &weights.at(&block),
                    &input.at(&block),
                    plan,
                    config,
                );
            }
        }
    }
}

/// One block of the cube's walk: this plane's rows, then this lane's columns and channel
/// lines, each output cell contracting its whole tap window at the leaf under `config`.
#[cube]
fn interpolate_block<E: Float>(
    output: &Tile<E>,
    weights: &Tile<E>,
    input: &Tile<E>,
    #[comptime] plan: InterpolateSpace,
    #[comptime] config: RegisterBlock,
) {
    for region in output
        .op_space(weights, input)
        .level(comptime!(plan.plane_level()))
    {
        let output_plane = output.at(&region);
        let weights_plane = weights.at(&region);
        let input_plane = input.at(&region);
        for cell in output_plane
            .op_space(&weights_plane, &input_plane)
            .level(comptime!(plan.lane_level()))
        {
            let mut output_cell = output_plane.at(&cell);
            output_cell.mm_with(
                &weights_plane.at(&cell),
                &input_plane.at(&cell),
                config,
                Semiring::SUM_PROD,
            );
        }
    }
}
