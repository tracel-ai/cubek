use super::map_coord;
use crate::definition::{BoundaryMode, Kernel, Resample};
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
        in_coord: &mut Sequence<i32>,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
    ) -> (Vector<F, N>, Vector<F, N>) {
        let input_shape = input.shape();

        let resampling_vectorized_axis = comptime!(is_resampling_vectorized_axis(
            config,
            vectorized_axis,
            N::value(),
        ));

        if resampling_vectorized_axis {
            resolve_vectorized_tap(
                tap_idx,
                input,
                &input_shape,
                out_coord,
                in_coord,
                config,
                vectorized_axis,
            )
        } else {
            resolve_scalar_tap(
                input,
                &input_shape,
                out_coord,
                in_coord,
                config,
                vectorized_axis,
            )
        }
    }
}

/// Check if vectorized axis is resampling axis.
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

/// Resolve taps for non-vectorized or non-resampling axis.
#[cube]
fn resolve_scalar_tap<F: Float, N: Size>(
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    input_shape: &CoordsDyn,
    out_coord: &CoordsDyn,
    in_coord: &mut Sequence<i32>,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let clamped_coord = clamp_to_coords_dyn(input_shape, in_coord);

    let weight = compute_weight::<F, N>(
        out_coord,
        in_coord,
        &clamped_coord,
        config,
        vectorized_axis,
        0_usize,
    );

    let value = input.read(clamped_coord);

    (value, Vector::new(weight))
}

/// Resolve taps for vectorized and resampling axis.
#[cube]
fn resolve_vectorized_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    input_shape: &CoordsDyn,
    out_coord: &CoordsDyn,
    in_coord: &mut Sequence<i32>,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let vector_size = N::value();

    let mut weight = Vector::empty();
    let mut value = Vector::empty();

    #[unroll]
    for lane in 0..vector_size {
        map_coord::<F>(tap_idx, out_coord, in_coord, lane, config, vectorized_axis);

        let clamped_coord = clamp_to_coords_dyn(input_shape, in_coord);

        let lane_weight = compute_weight::<F, N>(
            out_coord,
            in_coord,
            &clamped_coord,
            config,
            vectorized_axis,
            lane,
        );

        let extract_idx = clamped_coord[vectorized_axis] as usize % vector_size;

        let lane_values = input.read(clamped_coord);

        let lane_value = lane_values.extract(extract_idx);

        weight.insert(lane, lane_weight);
        value.insert(lane, lane_value);
    }

    (value, weight)
}

/// Clamp coordinates from Sequence<i32> to CoordsDyn, with bounds check.
#[cube]
pub fn clamp_to_coords_dyn(shape: &CoordsDyn, coords: &mut Sequence<i32>) -> CoordsDyn {
    let mut clamped_coord = CoordsDyn::new();

    #[unroll]
    for i in 0..coords.len() {
        clamped_coord.push(coords[i].clamp(0, (shape[i] - 1) as i32) as u32);
    }

    clamped_coord
}

/// Computes weight considering boundary mode.
#[cube]
fn compute_weight<F: Float, N: Size>(
    out_coord: &CoordsDyn,
    in_coord: &mut Sequence<i32>,
    clamped_coord: &CoordsDyn,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] lane: usize,
) -> F {
    match config.boundary {
        BoundaryMode::Clamp => {
            Kernel::weight::<F, N>(in_coord, out_coord, config, vectorized_axis, lane)
        }
        BoundaryMode::Zero => select(
            is_in_bounds(in_coord, clamped_coord),
            Kernel::weight::<F, N>(in_coord, out_coord, config, vectorized_axis, lane),
            F::zero(),
        ),
    }
}

/// Check if coordinate is in bounds depending on boundary mode.
#[cube]
fn is_in_bounds(in_coord: &mut Sequence<i32>, clamped_coord: &CoordsDyn) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for i in 0..in_coord.len() {
        in_bounds &= in_coord[i] == clamped_coord[i] as i32;
    }

    in_bounds
}
