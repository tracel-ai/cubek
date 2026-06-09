use crate::components::{
    footprint::get_footprint,
    kernel::{kernel_num_taps, kernel_weight},
    semiring::{semiring_combine, semiring_identity, semiring_reduce},
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
pub fn resample_kernel<C: Float>(
    input: &View<'_, C, CoordsDyn>,
    output: &mut ViewMut<'_, C, CoordsDyn>,
    output_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] config: Resample,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;

    if index >= working_units {
        terminate!();
    }

    let out_coord = get_coord(index, &output_shape);

    resample_coord::<C>(input, output, &out_coord, config);
}

/// Convert a linear index to a coordinate.
#[cube]
fn get_coord(index: usize, shape: &Sequence<FastDivmod<usize>>) -> CoordsDyn {
    let mut coords = CoordsDyn::new();
    let mut index = index;

    #[unroll]
    for i in 0..shape.len() {
        let (new_index, coord) = shape[i].div_mod(index);
        coords.push(coord as u32);
        index = new_index;
    }

    coords
}

/// Resample a single output coord.
#[cube]
pub fn resample_coord<C: Float>(
    input: &View<'_, C, CoordsDyn>,
    output: &mut ViewMut<'_, C, CoordsDyn>,
    out_coord: &CoordsDyn,
    #[comptime] config: Resample,
) {
    let mut acc = semiring_identity::<C>(&config.semiring);

    accumulate_taps::<C>(input, out_coord, &mut acc, config);

    output.write(out_coord.clone(), acc);
}

// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<C: Float>(
    input: &View<'_, C, CoordsDyn>,
    out_coord: &CoordsDyn,
    acc: &mut C,
    #[comptime] config: Resample,
) {
    let num_taps = kernel_num_taps(&config.kernel);
    let radius = (num_taps + 1) / 2;
    let num_axes = config.reduce_axes.len();

    #[unroll]
    for i in 0..num_taps {
        let flat_idx = RuntimeCell::<usize>::new(i);
        let mut weight_nd = C::cast_from(1.0);
        let mut in_coord = out_coord.clone();

        #[unroll]
        for dim in 0..num_axes {
            let axis = config.reduce_axes.index(dim);

            let out_pos = out_coord[*axis] as usize;

            let footprint = get_footprint::<C>(&config, radius, out_pos);

            let tap_1d_idx = flat_idx.read() % (num_taps);
            flat_idx.store(flat_idx.read() / (num_taps));

            let tap_pos = footprint.start_tap + tap_1d_idx as isize;
            let x = C::cast_from(tap_1d_idx as isize - radius as isize) - footprint.frac;
            let weight_1d = kernel_weight::<C>(&config.kernel, x);
            weight_nd *= weight_1d;

            in_coord[*axis] = tap_pos as u32;
        }

        if input.is_in_bounds(in_coord.clone()) {
            let value = input.read(in_coord);

            let combined = semiring_combine::<C>(&config.semiring, value, weight_nd);

            *acc = semiring_reduce::<C>(&config.semiring, *acc, combined);
        }
    }
}
