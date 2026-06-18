use crate::{
    components::coordinates::resolve_tap_coords,
    definition::{BoundaryMode, CoordsDynI, Kernel, Resample, ResampleArgs},
};
use cubecl::{prelude::*, std::tensor::View};

pub struct TapResolver;

#[cube]
impl TapResolver {
    #[allow(clippy::too_many_arguments)]
    pub fn resolve<F: Float, N: Size>(
        tap_idx: usize,
        input: &View<'_, Vector<F, N>, CoordsDynI>,
        out_coord: &CoordsDynI,
        start_coords: &CoordsDynI,
        centers: &Sequence<F>,
        args: &ResampleArgs,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] num_lanes: usize,
        #[comptime] vector_size: usize,
    ) -> (Vector<F, N>, Vector<F, N>) {
        let mut weight = Vector::empty();
        let mut value = Vector::empty();

        let input_shape = input.shape();

        for lane in comptime!(0..num_lanes) {
            let (in_coord, clamped_coord) = resolve_tap_coords(
                tap_idx,
                out_coord,
                &input_shape,
                start_coords,
                args,
                config,
                lane,
            );

            let unclamped_weight = Kernel::weight::<F, N>(&in_coord, centers, config, lane);

            let lane_weight = BoundaryMode::resolve_weight::<F, N>(
                unclamped_weight,
                &in_coord,
                &clamped_coord,
                &config.boundary,
            );

            // Extract input values only when vectorized over multiple lanes
            if num_lanes > 1 {
                let extract_idx = clamped_coord[vectorized_axis] as usize % vector_size;

                let lane_values = input.read(clamped_coord);
                let lane_value = lane_values.extract(extract_idx);

                weight.insert(lane, lane_weight);
                value.insert(lane, lane_value);
            } else {
                value = input.read(clamped_coord);
                weight = Vector::new(lane_weight);
            }
        }

        (value, weight)
    }
}
