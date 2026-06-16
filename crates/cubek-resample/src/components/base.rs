use crate::components::resample_instruction::Accumulator;
use crate::definition::{Resample, ResampleArgs};
use crate::{
    components::{resample_instruction::ResampleInstruction, tap_resolver::TapResolver},
    definition::Kernel,
};
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

    let out_coord = get_coord(index * output.vector_size(), &output_shape, &output_strides);

    resample_coord::<F, N>(input, output, &out_coord, &args, &config, vectorized_axis);
}

/// Convert a linear index to a coordinate.
#[cube]
fn get_coord(
    index: usize,
    shape: &Sequence<FastDivmod<usize>>,
    strides: &Sequence<FastDivmod<usize>>,
) -> CoordsDyn {
    let mut coords = CoordsDyn::new();

    #[unroll]
    for i in 0..shape.len() {
        let (index_at_dim, _) = strides[i].div_mod(index);

        let (_, coord) = shape[i].div_mod(index_at_dim);

        coords.push(coord as u32);
    }

    coords
}

/// Resample a single output coord.
#[cube]
pub fn resample_coord<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    output: &mut ViewMut<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) {
    let mut accumulator = ResampleInstruction::initialize(config);

    accumulate_taps::<F, N>(
        input,
        out_coord,
        &mut accumulator,
        args,
        config,
        vectorized_axis,
    );

    ResampleInstruction::store(out_coord.clone(), output, accumulator, config);
}

/// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    accumulator: &mut Accumulator<F, N>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) {
    let num_taps = comptime! {
        let mut num_taps = 1;
        for axis_idx in comptime!(0..config.resample_axes.len()) {
            let resample_axis = config.resample_axes.index(axis_idx);
            num_taps *= Kernel::num_taps(&resample_axis.kernel)
        }
        num_taps
    };

    let vector_size = N::value();

    let resampling_vectorized_axis = comptime!(is_resampling_vectorized_axis(
        config,
        vectorized_axis,
        vector_size,
    ));

    let (start_coords, centers) = compute_anchors::<F, N>(
        out_coord,
        args,
        config,
        vectorized_axis,
        resampling_vectorized_axis,
        vector_size,
    );

    #[unroll]
    for tap_idx in 0..num_taps {
        ResampleInstruction::count_position(accumulator, out_coord, config);

        let (mut value, weight) = TapResolver::resolve(
            tap_idx,
            input,
            out_coord,
            &start_coords,
            &centers,
            config,
            vectorized_axis,
            resampling_vectorized_axis,
        );

        ResampleInstruction::combine(&mut value, weight, tap_idx, config);

        ResampleInstruction::accumulate(accumulator, value, weight, tap_idx, config);
    }
}

/// Check if vectorized axis is resampling axis.
fn is_resampling_vectorized_axis(
    config: &Resample,
    vectorized_axis: usize,
    vector_size: usize,
) -> bool {
    let mut is_vectorized = false;

    for axis in comptime!(0..config.resample_axes.len()) {
        let resample_axis = config.resample_axes.index(axis);
        is_vectorized |= resample_axis.axis == vectorized_axis;
    }

    is_vectorized && vector_size > 1
}

/// Precompute the starting input coordinates and continuous centers.
#[cube]
pub fn compute_anchors<F: Float, N: Size>(
    out_coord: &CoordsDyn,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] resampling_vectorized_axis: bool,
    #[comptime] vector_size: usize,
) -> (Sequence<i32>, Sequence<F>) {
    let mut start_coords = Sequence::new();
    let mut centers = Sequence::<F>::new();

    let compute_lanes = comptime! {if resampling_vectorized_axis {vector_size} else {1}};

    #[unroll]
    for lane in 0..compute_lanes {
        #[unroll]
        for axis_idx in comptime!(0..config.resample_axes.len()) {
            let resample_axis = config.resample_axes.index(axis_idx);
            let resample_axis_args = args.resample_axes.index(axis_idx);

            let num_taps = Kernel::num_taps(&resample_axis.kernel);
            let radius = num_taps.div_ceil(2);

            let out_pos = out_coord[resample_axis.axis] as usize;

            let lane_out_pos = if resample_axis.axis == vectorized_axis {
                out_pos + lane
            } else {
                out_pos
            };

            let center = resample_axis_args
                .placement_args
                .map::<F>(lane_out_pos, &resample_axis.placement);

            let center_floored = center.floor();

            let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

            start_coords.push(start_tap as i32);
            centers.push(center);
        }
    }

    (start_coords, centers)
}
