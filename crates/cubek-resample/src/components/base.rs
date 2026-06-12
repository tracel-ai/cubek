use crate::components::resample_instruction::Accumulator;
use crate::definition::Resample;
use crate::{
    components::{
        resample_instruction::ResampleInstruction,
        tap_resolver::{SeparableTapResolver, TapResolver},
    },
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
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;

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
    let num_axes = config.resample_axes.len();

    accumulate_taps::<F, N>(
        input,
        out_coord,
        &mut accumulator,
        config,
        vectorized_axis,
        vector_size,
        num_axes,
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
    #[comptime] num_axes: usize,
) {
    let mut num_taps = 1;
    #[unroll]
    for axis_idx in 0..num_axes {
        let resample_axis = config.resample_axes.index(axis_idx);
        num_taps *= Kernel::num_taps(&resample_axis.kernel)
    }

    let mut in_coord = out_coord.clone();

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
            num_axes,
        );
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
    #[comptime] num_axes: usize,
) {
    let (mut value, weight) = <SeparableTapResolver as TapResolver<F, N>>::resolve(
        tap_idx,
        input,
        out_coord,
        in_coord,
        config,
        vectorized_axis,
        vector_size,
        num_axes,
    );

    if input.is_in_bounds(in_coord.clone()) {
        ResampleInstruction::count_position(accumulator, &out_coord, config);
    }

    ResampleInstruction::combine(&mut value, weight, tap_idx, config);

    ResampleInstruction::accumulate(accumulator, value, tap_idx, config);
}
