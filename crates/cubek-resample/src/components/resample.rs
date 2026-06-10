use crate::components::{
    footprint::Footprint,
    kernel::kernel_weight,
    semiring::{semiring_combine, semiring_combine_vec, semiring_identity, semiring_reduce},
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

    accumulate_taps::<F, N>(input, out_coord, &mut acc, config, vectorized_axis);

    output.write(out_coord.clone(), acc);
}

// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    acc: &mut Vector<F, N>,
    #[comptime] config: Resample,
    #[comptime] vectorized_axis: usize,
) {
    let mut footprints = Sequence::<Footprint<F>>::new();

    let num_axes = comptime!(config.resample_axes.len());

    let mut total_taps = 1;
    #[unroll]
    for dim in 0..num_axes {
        let resample_axis = config.resample_axes.index(dim);
        footprints.push(Footprint::new(resample_axis));
        total_taps *= Footprint::<F>::num_taps(resample_axis);
    }

    let vector_size = N::value();

    if vector_size > 1 {
        for i in 0..total_taps {
            let mut combined_vec = semiring_identity::<F, N>(&config.semiring);

            #[unroll]
            for v in 0..vector_size {
                let mut in_coord = out_coord.clone();
                let mut weight_nd_v = F::new(1.0);
                let mut current_flat_idx = i;

                #[unroll]
                for dim in 0..num_axes {
                    let resample_axis = config.resample_axes.index(dim);
                    let footprint = footprints.index(dim);

                    let num_taps = Footprint::<F>::num_taps(resample_axis);
                    let radius = (num_taps + 1) / 2;

                    let out_pos = out_coord[resample_axis.axis] as usize;

                    let lane_out_pos = if comptime!(resample_axis.axis == vectorized_axis) {
                        out_pos + v
                    } else {
                        out_pos
                    };

                    let (start_tap, frac) = footprint.start_tap_and_frac(radius, lane_out_pos);

                    let tap_1d_idx = current_flat_idx % num_taps;
                    current_flat_idx /= num_taps;

                    let tap_pos = start_tap + tap_1d_idx as isize;
                    let x = F::cast_from(tap_1d_idx as isize - radius as isize) - frac;

                    weight_nd_v *= kernel_weight::<F>(&resample_axis.kernel, x);
                    in_coord[resample_axis.axis] = tap_pos as u32;
                }

                if input.is_in_bounds(in_coord.clone()) {
                    let value_vec = input.read(in_coord.clone());
                    let val_v = value_vec.extract(v);
                    let combined_v = semiring_combine::<F>(&config.semiring, val_v, weight_nd_v);
                    combined_vec.insert(v, combined_v);
                }
            }

            *acc = semiring_reduce(&config.semiring, *acc, combined_vec);
        }
    } else {
        for i in 0..total_taps {
            let flat_idx = RuntimeCell::<usize>::new(i);
            let mut weight_nd = F::new(1.0);
            let mut in_coord = out_coord.clone();

            #[unroll]
            for dim in 0..num_axes {
                let resample_axis = config.resample_axes.index(dim);
                let footprint = footprints.index(dim);

                let num_taps = Footprint::<F>::num_taps(resample_axis);
                let radius = (num_taps + 1) / 2;

                let out_pos = out_coord[resample_axis.axis] as usize;

                let (start_tap, frac) = footprint.start_tap_and_frac(radius, out_pos);

                let tap_1d_idx = flat_idx.read() % (num_taps);
                flat_idx.store(flat_idx.read() / (num_taps));

                let tap_pos = start_tap + tap_1d_idx as isize;
                let x = F::cast_from(tap_1d_idx as isize - radius as isize) - frac;
                let weight_1d = kernel_weight::<F>(&resample_axis.kernel, x);
                weight_nd *= weight_1d;

                in_coord[resample_axis.axis] = tap_pos as u32;
            }

            if input.is_in_bounds(in_coord.clone()) {
                let value = input.read(in_coord);

                let combined =
                    semiring_combine_vec(&config.semiring, value, Vector::new(weight_nd));

                *acc = semiring_reduce(&config.semiring, *acc, combined);
            }
        }
    }
}
