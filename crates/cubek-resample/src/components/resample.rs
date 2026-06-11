use crate::components::{
    footprint::Footprint,
    line_combiner::{LineCombiner, ScalarLineCombiner, VectorizedLineCombiner},
    semiring::{semiring_identity, semiring_reduce},
    tap_resolver::TapResolver,
};
use crate::definition::Resample;
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

    resample_coord(input, output, &out_coord, config, vectorized_axis);
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
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
) {
    let mut acc = semiring_identity::<F, N>(&config.semiring);

    let vector_size = N::value();

    if comptime!(vector_size > 1) {
        accumulate_taps::<F, N, VectorizedLineCombiner>(
            input,
            out_coord,
            &mut acc,
            config,
            vectorized_axis,
            vector_size,
        );
    } else {
        accumulate_taps::<F, N, ScalarLineCombiner>(
            input,
            out_coord,
            &mut acc,
            config,
            vectorized_axis,
            vector_size,
        );
    }

    output.write(out_coord.clone(), acc);
}

// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<F: Float, N: Size, L: LineCombiner<F, N>>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    acc: &mut Vector<F, N>,
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) {
    let num_axes = config.resample_axes.len();

    let (footprints, total_taps) = build_footprints::<F>(&config, num_axes);

    let mut in_coord = out_coord.clone();

    for i in 0..total_taps {
        let mut combined = L::init_combined(&config);

        #[unroll]
        for lane in 0..vector_size {
            let weight = TapResolver::resolve::<F>(
                i,
                out_coord,
                &mut in_coord,
                lane,
                &footprints,
                &config,
                vectorized_axis,
                num_axes,
            );

            if input.is_in_bounds(in_coord.clone()) {
                let value = input.read(in_coord.clone());

                combined = L::process_lane(
                    &in_coord,
                    combined,
                    lane,
                    value,
                    weight,
                    &config,
                    vectorized_axis,
                    vector_size,
                );
            }
        }

        *acc = semiring_reduce(&config.semiring, *acc, combined);
    }
}

#[cube]
fn build_footprints<F: Float>(
    #[comptime] config: &Resample,
    #[comptime] num_axes: usize,
) -> (Sequence<Footprint<F>>, usize) {
    let mut footprints = Sequence::<Footprint<F>>::new();
    let mut total_taps = 1;

    #[unroll]
    for dim in 0..num_axes {
        let resample_axis = config.resample_axes.index(dim);
        footprints.push(Footprint::new(resample_axis));
        total_taps *= Footprint::<F>::num_taps(resample_axis);
    }

    (footprints, total_taps)
}
