use crate::components::coordinates::{compute_anchors, cube_absolute_coord, tile_absolute_coord};
use crate::components::{resample_instruction::ResampleInstruction, tap_resolver::TapResolver};
use crate::definition::{Accumulator, Resample, ResampleArgs, in_bounds};
use cubecl::std::{FastDivmod, FastDivmodExpand};
use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::CoordsDynI},
};

/// Resample kernel.
#[cube(launch_unchecked)]
pub fn resample_kernel<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDynI>,
    output: &mut ViewMut<'_, Vector<F, N>, CoordsDynI>,
    tile_shape: Sequence<FastDivmod<usize>>,
    tile_strides: Sequence<FastDivmod<usize>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    cube_strides: Sequence<FastDivmod<usize>>,
    args: ResampleArgs,
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[define(F)] _dtype: StorageType,
) {
    let vector_size = N::value();

    let cube_pos = CUBE_POS;

    let cube_coord = cube_absolute_coord(&cube_shape, &cube_strides, cube_pos);

    let unit_pos = UNIT_POS as usize;
    let cube_dim = CUBE_DIM as usize;

    let num_iterations = (sequence_area(&tile_shape) - unit_pos).div_ceil(cube_dim);

    for iteration in 0..num_iterations {
        let unit_pos = unit_pos + iteration * cube_dim;

        resample_unit(
            input,
            output,
            &cube_coord,
            unit_pos,
            &tile_shape,
            &tile_strides,
            &args,
            &config,
            vectorized_axis,
            vector_size,
        );
    }
}

/// Resample a single unit (a slice of the output).
#[cube]
fn resample_unit<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDynI>,
    output: &mut ViewMut<'_, Vector<F, N>, CoordsDynI>,
    cube_coord: &CoordsDynI,
    unit_pos: usize,
    tile_shape: &Sequence<FastDivmod<usize>>,
    tile_strides: &Sequence<FastDivmod<usize>>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    let out_coord = tile_absolute_coord(
        tile_shape,
        tile_strides,
        cube_coord,
        unit_pos,
        vectorized_axis,
        vector_size,
    );

    if in_bounds(&out_coord, &output.shape(), config) {
        let mut accumulator = ResampleInstruction::initialize(config);

        accumulate_taps::<F, N>(
            input,
            &out_coord,
            &mut accumulator,
            args,
            config,
            vectorized_axis,
            vector_size,
        );

        ResampleInstruction::store(out_coord, output, &accumulator, config);
    }
}

/// Accumulate taps to produce a single output value.
#[cube]
fn accumulate_taps<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDynI>,
    out_coord: &CoordsDynI,
    accumulator: &mut Accumulator<F, N>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    let num_lanes = config.compute_num_lanes(vectorized_axis, vector_size);

    let (start_coords, centers) =
        compute_anchors::<F>(out_coord, args, config, vectorized_axis, num_lanes);

    let num_taps = Resample::calculate_num_taps(args, config);

    for tap_idx in 0..num_taps {
        accumulate_tap(
            tap_idx,
            input,
            out_coord,
            &start_coords,
            &centers,
            accumulator,
            args,
            config,
            vectorized_axis,
            num_lanes,
            vector_size,
        );
    }
}

/// Accumulate a single tap to produce a single output value.
#[cube]
fn accumulate_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDynI>,
    out_coord: &CoordsDynI,
    start_coords: &CoordsDynI,
    centers: &Sequence<F>,
    accumulator: &mut Accumulator<F, N>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] num_lanes: usize,
    #[comptime] vector_size: usize,
) {
    ResampleInstruction::count_position(accumulator, out_coord, config);

    let (mut value, weight) = TapResolver::resolve(
        tap_idx,
        input,
        out_coord,
        start_coords,
        centers,
        args,
        config,
        vectorized_axis,
        num_lanes,
        vector_size,
    );

    ResampleInstruction::combine(&mut value, weight, tap_idx, config);

    ResampleInstruction::accumulate(accumulator, value, weight, tap_idx, config);
}

/// Compute the area of a sequence of FastDivmod.
#[cube]
pub fn sequence_area(shape: &Sequence<FastDivmod<usize>>) -> usize {
    let mut area = 1;

    #[unroll]
    for i in 0..shape.len() {
        area *= fast_div_mod_value(&shape[i]);
    }

    area
}

/// Get the value of a FastDivmod.
#[cube]
pub fn fast_div_mod_value(div_mod: &FastDivmod<usize>) -> usize {
    match div_mod {
        FastDivmod::Fast { divisor, .. } => *divisor,
        FastDivmod::Fallback { divisor } => *divisor,
    }
}
