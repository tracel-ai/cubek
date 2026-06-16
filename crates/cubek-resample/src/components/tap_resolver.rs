use crate::{
    components::coordinates::{CoordsDynI, is_in_bounds, resolve_tap_coords},
    definition::{BoundaryMode, Kernel, Resample, ResampleArgs},
};
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

            let lane_weight =
                compute_weight::<F, N>(&in_coord, &clamped_coord, centers, config, lane);

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

/// Computes weight considering boundary mode.
#[cube]
fn compute_weight<F: Float, N: Size>(
    in_coord: &CoordsDynI,
    clamped_coord: &CoordsDyn,
    centers: &Sequence<F>,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) -> F {
    match config.boundary {
        BoundaryMode::Clamp => Kernel::weight::<F, N>(in_coord, centers, config, lane),
        BoundaryMode::Zero => select(
            is_in_bounds(in_coord, &clamped_coord),
            Kernel::weight::<F, N>(in_coord, centers, config, lane),
            F::zero(),
        ),
    }
}
