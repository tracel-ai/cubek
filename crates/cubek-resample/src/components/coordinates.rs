use crate::definition::{Resample, ResampleArgs};
use cubecl::{
    prelude::*,
    std::{FastDivmod, tensor::layout::CoordsDyn},
};

pub type CoordsDynI = Sequence<i32>;

/// Convert a linear index to a coordinate.
#[cube]
pub fn coord_from_index(
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

/// Precompute the starting input coordinates and continuous centers.
#[cube]
pub fn compute_anchors<F: Float, N: Size>(
    out_coord: &CoordsDyn,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] num_lanes: usize,
) -> (CoordsDynI, Sequence<F>) {
    let mut start_coords = CoordsDynI::new();
    let mut centers = Sequence::<F>::new();

    #[unroll]
    for lane in 0..num_lanes {
        #[unroll]
        for axis_idx in comptime!(0..config.resample_axes.len()) {
            let resample_axis = config.resample_axes.index(axis_idx);
            let resample_axis_args = args.resample_axes.index(axis_idx);

            let radius = resample_axis_args.window_args.size.div_ceil(2);

            let out_pos = out_coord[resample_axis.axis] as usize;

            let lane_out_pos = if resample_axis.axis == vectorized_axis {
                out_pos + lane
            } else {
                out_pos
            };

            let center = resample_axis_args
                .placement_args
                .map::<F>(lane_out_pos, &resample_axis.placement);

            let center_floored = center.floor();

            let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

            start_coords.push(start_tap as i32);
            centers.push(center);
        }
    }

    (start_coords, centers)
}

/// Helper to map and clamp coordinates for a specific lane.
#[cube]
pub fn resolve_tap_coords(
    tap_idx: usize,
    out_coord: &CoordsDyn,
    input_shape: &CoordsDyn,
    start_coords: &CoordsDynI,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) -> (CoordsDynI, CoordsDyn) {
    let mut in_coord = from_coords_dyn(out_coord);

    map_coord(tap_idx, &mut in_coord, start_coords, args, config, lane);

    let clamped_coord = clamp_to_coords_dyn(input_shape, &in_coord);

    (in_coord, clamped_coord)
}

/// Check if coordinate is in bounds depending on boundary mode.
#[cube]
pub fn is_in_bounds(in_coord: &CoordsDynI, clamped_coord: &CoordsDyn) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for i in 0..in_coord.len() {
        in_bounds &= in_coord[i] == clamped_coord[i] as i32;
    }

    in_bounds
}

/// Map output coordinate to input coordinate using precomputed anchors.
#[cube]
fn map_coord(
    tap_idx: usize,
    in_coord: &mut CoordsDynI,
    start_coords: &CoordsDynI,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) {
    let mut current_flat_idx = tap_idx;
    let num_axes = comptime!(config.resample_axes.len());

    #[unroll]
    for axis_idx in comptime!(0..num_axes) {
        let resample_axis = config.resample_axes.index(axis_idx);
        let resample_axis_args = args.resample_axes.index(axis_idx);

        let tap_axis_idx = current_flat_idx % resample_axis_args.window_args.size;
        current_flat_idx /= resample_axis_args.window_args.size;

        let flat_idx = lane * num_axes + axis_idx;

        in_coord[resample_axis.axis] = start_coords[flat_idx] + tap_axis_idx as i32;
    }
}

/// Convert CoordsDyn to CoordsDynI.
#[cube]
fn from_coords_dyn(coords: &CoordsDyn) -> CoordsDynI {
    let mut coords_i32 = Sequence::new();

    #[unroll]
    for i in 0..coords.len() {
        coords_i32.push(coords[i] as i32);
    }

    coords_i32
}

/// Clamp coordinates from CoordsDynI to CoordsDyn, with bounds check.
#[cube]
fn clamp_to_coords_dyn(shape: &CoordsDyn, coords: &CoordsDynI) -> CoordsDyn {
    let mut clamped_coord = CoordsDyn::new();

    #[unroll]
    for i in 0..coords.len() {
        clamped_coord.push(coords[i].clamp(0, (shape[i] - 1) as i32) as u32);
    }

    clamped_coord
}
