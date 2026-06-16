use crate::components::coordinates::{compute_anchors, coord_from_index};
use crate::components::resample_instruction::Accumulator;
use crate::components::{resample_instruction::ResampleInstruction, tap_resolver::TapResolver};
use crate::definition::{Resample, ResampleArgs};
use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::{View, ViewMut, layout::CoordsDyn},
    },
};

/// Resample kernel.
#[cube(launch_unchecked)]
pub fn resample_kernel<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    output: &mut ViewMut<'_, Vector<F, N>, CoordsDyn>,
    output_shape: Sequence<FastDivmod<usize>>,
    output_strides: Sequence<FastDivmod<usize>>,
    working_units: usize,
    args: ResampleArgs,
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;

    if index >= working_units {
        terminate!();
    }

    let vector_size = N::value();

    let out_coord = coord_from_index(index * vector_size, &output_shape, &output_strides);

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

    ResampleInstruction::store(out_coord.clone(), output, accumulator, &config);
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
    let num_lanes = comptime!(compute_num_lanes(config, vectorized_axis, vector_size));

    let (start_coords, centers) =
        compute_anchors::<F, N>(out_coord, args, config, vectorized_axis, num_lanes);

    let num_taps = compute_num_taps(args, config);

    for tap_idx in 0..num_taps {
        ResampleInstruction::count_position(accumulator, out_coord, config);

        let (mut value, weight) = TapResolver::resolve(
            tap_idx,
            input,
            out_coord,
            &start_coords,
            &centers,
            args,
            config,
            vectorized_axis,
            num_lanes,
            vector_size,
        );

        ResampleInstruction::combine(&mut value, weight, tap_idx, config);

        ResampleInstruction::accumulate(accumulator, value, weight, tap_idx, config);
    }
}

/// Returns the number of lanes to unroll: `vector_size` if the vectorized axis
/// is a resampling axis, otherwise `1`.
fn compute_num_lanes(config: &Resample, vectorized_axis: usize, vector_size: usize) -> usize {
    let mut is_vectorized = false;

    for axis_idx in comptime!(0..config.resample_axes.len()) {
        is_vectorized |= config.resample_axes[axis_idx].axis == vectorized_axis;
    }

    if is_vectorized { vector_size } else { 1 }
}

/// Computes the total number of taps.
#[cube]
fn compute_num_taps(args: &ResampleArgs, #[comptime] config: &Resample) -> usize {
    let mut num_taps = 1;

    for axis_idx in comptime!(0..config.resample_axes.len()) {
        num_taps *= args.resample_axes.index(axis_idx).window_args.size
    }

    num_taps
}
