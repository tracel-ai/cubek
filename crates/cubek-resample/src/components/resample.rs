use crate::components::{
    footprint::{Footprint, get_footprint},
    kernel::kernel_weight,
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
    #[comptime] spatial_axis: usize,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;

    if index >= working_units {
        terminate!();
    }

    let out_coord = get_coord(index, &output_shape);

    resample_coord::<C>(input, output, &out_coord, config, spatial_axis);
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
    #[comptime] spatial_axis: usize,
) {
    let out_pos = out_coord[spatial_axis] as usize;

    let footprint = get_footprint::<C>(config, out_pos);

    let mut acc = semiring_identity::<C>(config.semiring);

    accumulate_taps::<C>(input, out_coord, footprint, &mut acc, config, spatial_axis);

    output.write(out_coord.clone(), acc);
}

// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<C: Float>(
    input: &View<'_, C, CoordsDyn>,
    out_coord: &CoordsDyn,
    footprint: Footprint<C>,
    acc: &mut C,
    #[comptime] config: Resample,
    #[comptime] spatial_axis: usize,
) {
    for i in 0..footprint.num_taps {
        let tap_pos = footprint.start_tap + i as isize;

        let x = C::cast_from(i as isize - footprint.radius as isize) - footprint.frac;
        let weight = kernel_weight::<C>(config.kernel, x);

        let mut in_coord = out_coord.clone();
        in_coord[spatial_axis] = tap_pos as u32;

        if input.is_in_bounds(in_coord.clone()) {
            let value = input.read(in_coord);

            let combined = semiring_combine::<C>(config.semiring, value, weight);

            *acc = semiring_reduce::<C>(config.semiring, *acc, combined);
        }
    }
}
