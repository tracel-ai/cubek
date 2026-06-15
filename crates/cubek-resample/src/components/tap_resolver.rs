use super::map_coord;
use crate::definition::{Kernel, Resample};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::CoordsDyn},
};

pub struct TapResolver;

#[cube]
impl TapResolver {
    #[allow(clippy::type_complexity)]
    pub fn resolve<F: Float, N: Size>(
        tap_idx: usize,
        input: &View<'_, Vector<F, N>, CoordsDyn>,
        out_coord: &CoordsDyn,
        in_coord: &mut CoordsDyn,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
    ) -> (Vector<F, N>, Vector<F, N>) {
        let resampling_vectorized_axis = comptime!(is_resampling_vectorized_axis(
            config,
            vectorized_axis,
            vector_size,
        ));

        if resampling_vectorized_axis {
            resolve_vectorized_tap(
                tap_idx,
                input,
                out_coord,
                in_coord,
                config,
                vectorized_axis,
                vector_size,
            )
        } else {
            resolve_scalar_tap(input, out_coord, in_coord, config, vectorized_axis)
        }
    }
}

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

#[cube]
fn resolve_scalar_tap<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    in_coord: &mut CoordsDyn,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let weight = Vector::new(Kernel::weight::<F>(
        in_coord,
        out_coord,
        config,
        vectorized_axis,
        0_usize,
    ));

    let value = input.read(in_coord.clone());

    (value, weight)
}

#[cube]
fn resolve_vectorized_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    out_coord: &CoordsDyn,
    in_coord: &mut CoordsDyn,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let mut weight = Vector::empty();
    let mut value = Vector::empty();

    #[unroll]
    for lane in 0..vector_size {
        map_coord::<F>(tap_idx, out_coord, in_coord, lane, config, vectorized_axis);

        let lane_weight = Kernel::weight::<F>(in_coord, out_coord, config, vectorized_axis, lane);

        let lane_values = input.read(in_coord.clone());
        let extract_idx = in_coord[vectorized_axis] as usize % vector_size;
        let lane_value = lane_values.extract(extract_idx);

        weight.insert(lane, lane_weight);
        value.insert(lane, lane_value);
    }

    (value, weight)
}
