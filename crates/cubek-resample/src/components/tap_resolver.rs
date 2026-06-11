use crate::{
    components::{footprint::Footprint, kernel::kernel_weight},
    definition::Resample,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};
use std::hash::Hash;

#[cube]
pub trait TapResolver<F: Float>: Send + Sync + 'static {
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    fn resolve<N: Size>(
        tap_idx: usize,
        out_coord: &CoordsDyn,
        in_coord: &mut CoordsDyn,
        footprints: &Sequence<Footprint<F>>,
        #[comptime] config: &Self::Config,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
        #[comptime] num_axes: usize,
    ) -> Vector<F, N>;
}

pub struct SeparableTapResolver;

#[cube]
impl<F: Float> TapResolver<F> for SeparableTapResolver {
    type Config = Resample;

    fn resolve<N: Size>(
        tap_idx: usize,
        out_coord: &CoordsDyn,
        in_coord: &mut CoordsDyn,
        footprints: &Sequence<Footprint<F>>,
        #[comptime] config: &Self::Config,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
        #[comptime] num_axes: usize,
    ) -> Vector<F, N> {
        let resampling_vectorized_axis = comptime!(is_resampling_vectorized_axis(
            config,
            vectorized_axis,
            vector_size,
            num_axes
        ));

        if resampling_vectorized_axis {
            let mut weights = Vector::empty();

            for i in 0..vector_size {
                // Reverse the order for the read to have the first tap index,
                // which correspond to the smallest input index.
                let lane = vector_size - 1 - i;

                let weight = compute_weight(
                    tap_idx,
                    out_coord,
                    in_coord,
                    lane,
                    footprints,
                    config,
                    vectorized_axis,
                    num_axes,
                );

                weights.insert(lane, weight);
            }

            weights
        } else {
            Vector::new(compute_weight(
                tap_idx,
                out_coord,
                in_coord,
                0,
                footprints,
                config,
                vectorized_axis,
                num_axes,
            ))
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
    footprints: &Sequence<Footprint<F>>,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] num_axes: usize,
) -> F {
    in_coord[vectorized_axis] = out_coord[vectorized_axis] + lane as u32;

    let mut weight = F::new(1.0);
    let mut current_flat_idx = tap_idx;

    #[unroll]
    for axis in 0..num_axes {
        let resample_axis = config.resample_axes.index(axis);
        let footprint = footprints.index(axis);

        let num_taps = Footprint::<F>::num_taps(resample_axis);
        let radius = (num_taps + 1) / 2;

        let out_pos = out_coord[resample_axis.axis] as usize;

        let lane_out_pos = if resample_axis.axis == vectorized_axis {
            out_pos + lane
        } else {
            out_pos
        };

        let (start_tap, frac) = footprint.start_tap_and_frac(radius, lane_out_pos);

        let tap_1d_idx = current_flat_idx % num_taps;
        current_flat_idx /= num_taps;

        let tap_pos = start_tap + tap_1d_idx as isize;
        let x = F::cast_from(tap_1d_idx as isize - radius as isize) - frac;

        weight *= kernel_weight::<F>(&resample_axis.kernel, x);
        in_coord[resample_axis.axis] = tap_pos as u32;
    }

    weight
}
