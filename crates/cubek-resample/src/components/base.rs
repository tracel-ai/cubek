use crate::components::resample_instruction::Accumulator;
use crate::definition::Resample;
use crate::{
    components::{resample_instruction::ResampleInstruction, tap_resolver::TapResolver},
    definition::{Kernel, Placement},
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
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;

    if index >= working_units {
        terminate!();
    }

    let out_coord = get_coord(index * output.vector_size(), &output_shape, &output_strides);

    resample_coord::<F, N>(input, output, &out_coord, &config, vectorized_axis);
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
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) {
    let mut accumulator = ResampleInstruction::initialize(config);

    let vector_size = N::value();

    accumulate_taps::<F, N>(
        input,
        out_coord,
        &mut accumulator,
        config,
        vectorized_axis,
        vector_size,
    );

    ResampleInstruction::store(out_coord.clone(), output, accumulator, config);
}

/// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    accumulator: &mut Accumulator<F, N>,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    let num_taps = comptime! {
        let mut num_taps = 1;
        for axis_idx in comptime!(0..config.resample_axes.len()) {
            let resample_axis = config.resample_axes.index(axis_idx);
            num_taps *= Kernel::num_taps(&resample_axis.kernel)
        }
        num_taps
    };

    let mut in_coord = out_coord.clone();

    #[unroll]
    for tap_idx in 0..num_taps {
        accumulate_tap::<F, N>(
            tap_idx,
            input,
            out_coord,
            &mut in_coord,
            accumulator,
            config,
            vectorized_axis,
            vector_size,
        );
    }
}

/// Map output coordinate to input coordinate for a given tap index and lane.
#[cube]
pub fn map_coord<F: Float>(
    tap_idx: usize,
    out_coord: &CoordsDyn,
    in_coord: &mut CoordsDyn,
    lane: usize,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) {
    in_coord[vectorized_axis] = out_coord[vectorized_axis] + lane as u32;

    let mut current_flat_idx = tap_idx;

    #[unroll]
    for axis_idx in comptime!(0..config.resample_axes.len()) {
        let resample_axis = config.resample_axes.index(axis_idx);

        let num_taps = Kernel::num_taps(&resample_axis.kernel);
        let radius = num_taps.div_ceil(2);

        let out_pos = out_coord[resample_axis.axis] as usize;

        let lane_out_pos = if resample_axis.axis == vectorized_axis {
            out_pos + lane
        } else {
            out_pos
        };

        let center = Placement::map::<F>(lane_out_pos, &resample_axis.placement);
        let center_floored = center.floor();

        let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

        let tap_1d_idx = current_flat_idx % num_taps;
        current_flat_idx /= num_taps;

        let tap_pos = start_tap + tap_1d_idx as isize;
        in_coord[resample_axis.axis] = tap_pos as u32;
    }
}

/// Accumulate taps for a single tap index.
#[cube]
fn accumulate_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    in_coord: &mut CoordsDyn,
    accumulator: &mut Accumulator<F, N>,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    map_coord::<F>(tap_idx, out_coord, in_coord, 0, config, vectorized_axis);

    ResampleInstruction::count_position(accumulator, out_coord, config);

    if input.is_in_bounds(in_coord.clone()) {
        let (mut value, weight) = TapResolver::resolve(
            tap_idx,
            input,
            out_coord,
            in_coord,
            config,
            vectorized_axis,
            vector_size,
        );

        ResampleInstruction::combine(&mut value, weight, tap_idx, config);

        ResampleInstruction::accumulate(accumulator, value, tap_idx, config);
    }
}
