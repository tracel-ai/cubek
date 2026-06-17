use crate::components::coordinates::{
    CoordsDynI, compute_anchors, cube_absolute_coord, tile_absolute_coord,
};
use crate::components::resample_instruction::Accumulator;
use crate::components::{resample_instruction::ResampleInstruction, tap_resolver::TapResolver};
use crate::definition::{Resample, ResampleArgs, TileArgs};
use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::CoordsDyn},
};

/// Resample kernel.
#[cube(launch_unchecked)]
pub fn resample_kernel<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    output: &mut ViewMut<'_, Vector<F, N>, CoordsDyn>,
    tile_args: TileArgs,
    args: ResampleArgs,
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[define(F)] _dtype: StorageType,
) {
    let vector_size = N::value();

    let cube_coord = cube_absolute_coord(
        &tile_args.cube_shape,
        &tile_args.cube_strides,
        vectorized_axis,
        vector_size,
    );

    let out_coord = tile_absolute_coord(
        &tile_args.tile_shape,
        &tile_args.tile_strides,
        &cube_coord,
        vectorized_axis,
        vector_size,
    );

    if is_out_of_bounds(&tile_args.output_shape, &out_coord, &config) {
        terminate!();
    }

    let mut accumulator = ResampleInstruction::initialize(&config);

    accumulate_taps::<F, N>(
        input,
        &out_coord,
        &mut accumulator,
        &args,
        &config,
        vectorized_axis,
        vector_size,
    );

    ResampleInstruction::store(out_coord, output, accumulator, &config);
}

/// Checks if the given coordinate is out of bounds for the given output shape.
#[cube]
fn is_out_of_bounds(
    output_shape: &Sequence<usize>,
    out_coord: &CoordsDyn,
    #[comptime] config: &Resample,
) -> bool {
    let mut is_out_of_bounds = false;

    for axis_idx in comptime!(0..config.resample_axes.len()) {
        if out_coord[axis_idx] as usize >= output_shape[axis_idx] {
            is_out_of_bounds = true;
        }
    }
    is_out_of_bounds
}

/// Accumulate taps to produce a single output value.
#[cube]
fn accumulate_taps<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    accumulator: &mut Accumulator<F, N>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    let num_lanes = config.compute_num_lanes(vectorized_axis, vector_size);

    let (start_coords, centers) =
        compute_anchors::<F>(out_coord, args, config, vectorized_axis, num_lanes);

    let num_taps = Resample::compute_num_taps(args, config);

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
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
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
