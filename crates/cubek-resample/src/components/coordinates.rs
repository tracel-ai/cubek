use crate::definition::{CoordsDynI, Resample, ResampleArgs, fast_div_mod_value};
use cubecl::{
    prelude::*,
    std::{FastDivmod, tensor::layout::CoordsDyn},
};

/// Computes the absolute coordinate of a cube.
#[cube]
pub fn cube_absolute_coord(
    cube_shape: &Sequence<FastDivmod<usize>>,
    cube_strides: &Sequence<FastDivmod<usize>>,
    cube_pos: usize,
) -> CoordsDyn {
    let mut coords = CoordsDyn::new();

    #[unroll]
    for i in 0..cube_shape.len() {
        let (cube_pos_at_dim, _) = cube_strides[i].div_mod(cube_pos);
        let (_, cube_coord) = cube_shape[i].div_mod(cube_pos_at_dim);

        coords.push(cube_coord as u32);
    }

    coords
}

/// Computes the local coordinate within a tile.
#[cube]
pub fn tile_absolute_coord(
    tile_shape: &Sequence<FastDivmod<usize>>,
    tile_strides: &Sequence<FastDivmod<usize>>,
    cube_coord: &CoordsDyn,
    unit_pos: usize,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) -> CoordsDyn {
    let mut coords = CoordsDyn::new();

    #[unroll]
    for i in 0..tile_shape.len() {
        let (unit_pos_at_dim, _) = tile_strides[i].div_mod(unit_pos);
        let (_, coord) = tile_shape[i].div_mod(unit_pos_at_dim);

        let tile_dim_size = fast_div_mod_value(&tile_shape[i]);

        let coord = if i == vectorized_axis {
            ((cube_coord[i] as usize * tile_dim_size + coord) * vector_size) as u32
        } else {
            (cube_coord[i] as usize * tile_dim_size + coord) as u32
        };

        coords.push(coord);
    }

    coords
}

/// Checks if the given coordinate is in bounds for the given output shape.
#[cube]
pub fn in_bounds(
    output_shape: &Sequence<usize>,
    out_coord: &CoordsDyn,
    #[comptime] config: &Resample,
) -> bool {
    let mut in_bounds = true;

    for axis_idx in comptime!(0..config.resample_axes.len()) {
        let resample_axis = config.resample_axes.index(axis_idx);
        let axis = resample_axis.axis;

        if out_coord[axis] as usize >= output_shape[axis] {
            in_bounds = false;
        }
    }
    in_bounds
}

/// Precompute the starting input coordinates and continuous centers.
#[cube]
pub fn compute_anchors<F: Float>(
    out_coord: &CoordsDyn,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] num_lanes: usize,
) -> (CoordsDynI, Sequence<F>) {
    let mut start_coords = CoordsDynI::new();
    let mut centers = Sequence::<F>::new();

    for lane in comptime!(0..num_lanes) {
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
    let mut flat_idx = tap_idx;
    let num_axes = comptime!(config.resample_axes.len());

    for axis_idx in comptime!(0..num_axes) {
        let resample_axis = config.resample_axes.index(axis_idx);
        let resample_axis_args = args.resample_axes.index(axis_idx);

        let tap_axis_idx = flat_idx % resample_axis_args.window_args.size;
        flat_idx /= resample_axis_args.window_args.size;

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
