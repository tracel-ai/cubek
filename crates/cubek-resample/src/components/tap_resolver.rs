use crate::definition::{Kernel, Placement, Resample};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::CoordsDyn},
};
use std::hash::Hash;

#[cube]
pub trait TapResolver<F: Float, N: Size>: Send + Sync + 'static {
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    fn resolve(
        tap_idx: usize,
        input: &View<'_, Vector<F, N>, CoordsDyn>,
        out_coord: &CoordsDyn,
        in_coord: &mut CoordsDyn,
        #[comptime] config: &Self::Config,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
        #[comptime] num_axes: usize,
    ) -> (Vector<F, N>, Vector<F, N>);
}

pub struct SeparableTapResolver;

#[cube]
impl<F: Float, N: Size> TapResolver<F, N> for SeparableTapResolver {
    type Config = Resample;

    #[allow(clippy::type_complexity)]
    fn resolve(
        tap_idx: usize,
        input: &View<'_, Vector<F, N>, CoordsDyn>,
        out_coord: &CoordsDyn,
        in_coord: &mut CoordsDyn,
        #[comptime] config: &Self::Config,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
        #[comptime] num_axes: usize,
    ) -> (Vector<F, N>, Vector<F, N>) {
        let resampling_vectorized_axis = comptime!(is_resampling_vectorized_axis(
            config,
            vectorized_axis,
            vector_size,
            num_axes
        ));

        if resampling_vectorized_axis {
            let mut weight = Vector::empty();
            let mut value = Vector::empty();

            for lane in 0..vector_size {
                let lane_weight = compute_weight::<F>(
                    tap_idx,
                    out_coord,
                    in_coord,
                    lane,
                    config,
                    vectorized_axis,
                    num_axes,
                );

                let lane_values = input.read(in_coord.clone());
                let extract_idx = in_coord[vectorized_axis] as usize % vector_size;
                let lane_value = lane_values.extract(extract_idx);

                weight.insert(lane, lane_weight);
                value.insert(lane, lane_value);
            }

            (value, weight)
        } else {
            let weight = Vector::new(compute_weight::<F>(
                tap_idx,
                out_coord,
                in_coord,
                0,
                config,
                vectorized_axis,
                num_axes,
            ));

            let value = input.read(in_coord.clone());

            (value, weight)
        }
    }
}

fn is_resampling_vectorized_axis(
    config: &Resample,
    vectorized_axis: usize,
    vector_size: usize,
    num_axes: usize,
) -> bool {
    let mut is_vectorized = false;

    for axis in 0..num_axes {
        let resample_axis = config.resample_axes.index(axis);
        is_vectorized |= resample_axis.axis == vectorized_axis;
    }

    is_vectorized && vector_size > 1
}

#[cube]
fn compute_weight<F: Float>(
    tap_idx: usize,
    out_coord: &CoordsDyn,
    in_coord: &mut CoordsDyn,
    lane: usize,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] num_axes: usize,
) -> F {
    in_coord[vectorized_axis] = out_coord[vectorized_axis] + lane as u32;

    let mut weight = F::new(1.0);
    let mut current_flat_idx = tap_idx;

    #[unroll]
    for axis_idx in 0..num_axes {
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

        let frac = center - center_floored;

        let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

        let tap_1d_idx = current_flat_idx % num_taps;
        current_flat_idx /= num_taps;

        let tap_pos = start_tap + tap_1d_idx as isize;
        let x = F::cast_from(tap_1d_idx as isize - radius as isize) - frac;

        weight *= Kernel::weight::<F>(x, &resample_axis.kernel);
        in_coord[resample_axis.axis] = tap_pos as u32;
    }

    weight
}
