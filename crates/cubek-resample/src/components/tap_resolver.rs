use crate::{
    components::{footprint::Footprint, kernel::kernel_weight},
    definition::Resample,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

pub struct TapResolver {}

#[cube]
impl TapResolver {
    pub fn resolve<F: Float>(
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
        for dim in 0..num_axes {
            let resample_axis = config.resample_axes.index(dim);
            let footprint = footprints.index(dim);

            let num_taps = Footprint::<F>::num_taps(resample_axis);
            let radius = (num_taps + 1) / 2;

            let out_pos = out_coord[resample_axis.axis] as usize;

            let lane_out_pos = if comptime!(resample_axis.axis == vectorized_axis) {
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
}
