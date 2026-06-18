use crate::definition::{CoordsDynI, Resample, ResampleArgs, TileSize, fast_div_mod_value};
use cubecl::prelude::*;

/// Computes the absolute coordinate of a cube.
#[cube]
pub fn cube_absolute_coord(cube_size: &TileSize, cube_pos: usize) -> CoordsDynI {
    let mut coords = CoordsDynI::new();

    #[unroll]
    for i in 0..cube_size.rank() {
        let (cube_pos_at_dim, _) = cube_size.strides[i].div_mod(cube_pos);
        let (_, cube_coord) = cube_size.shape[i].div_mod(cube_pos_at_dim);

        coords.push(cube_coord as i32);
    }

    coords
}

/// Computes the local coordinate within a tile.
#[cube]
pub fn tile_absolute_coord(
    tile_size: &TileSize,
    cube_coord: &CoordsDynI,
    unit_pos: usize,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) -> CoordsDynI {
    let mut coords = CoordsDynI::new();

    #[unroll]
    for i in 0..tile_size.rank() {
        let (unit_pos_at_dim, _) = tile_size.strides[i].div_mod(unit_pos);
        let (_, coord) = tile_size.shape[i].div_mod(unit_pos_at_dim);

        let tile_dim_size = fast_div_mod_value(&tile_size.shape[i]);

        let coord = if i == vectorized_axis {
            ((cube_coord[i] as usize * tile_dim_size + coord) * vector_size) as i32
        } else {
            (cube_coord[i] as usize * tile_dim_size + coord) as i32
        };

        coords.push(coord);
    }

    coords
}

/// Checks if the given coordinate is in bounds for the given output shape.
#[cube]
pub fn in_bounds(
    output_shape: &CoordsDynI,
    out_coord: &CoordsDynI,
    #[comptime] config: &Resample,
) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for axis_idx in 0..config.num_axes() {
        let resample_axis = config.resample_axes.index(axis_idx);
        let axis = resample_axis.axis;

        if out_coord[axis] >= output_shape[axis] {
            in_bounds = false;
        }
    }
    in_bounds
}

/// Precompute the starting input coordinates and continuous centers.
#[cube]
pub fn compute_anchors<F: Float>(
    out_coord: &CoordsDynI,
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
        for axis_idx in 0..config.num_axes() {
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
    out_coord: &CoordsDynI,
    input_shape: &CoordsDynI,
    start_coords: &CoordsDynI,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) -> (CoordsDynI, CoordsDynI) {
    let mut in_coord = out_coord.clone();

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

    #[unroll]
    for axis_idx in 0..config.num_axes() {
        let resample_axis = config.resample_axes.index(axis_idx);
        let resample_axis_args = args.resample_axes.index(axis_idx);

        let tap_axis_idx = flat_idx % resample_axis_args.window_args.size;
        flat_idx /= resample_axis_args.window_args.size;

        let flat_idx = lane * config.num_axes() + axis_idx;

        in_coord[resample_axis.axis] = start_coords[flat_idx] + tap_axis_idx as i32;
    }
}

/// Clamp coordinates from CoordsDynI to CoordsDyn, with bounds check.
#[cube]
fn clamp_to_coords_dyn(shape: &CoordsDynI, coords: &CoordsDynI) -> CoordsDynI {
    let mut clamped_coord = CoordsDynI::new();

    #[unroll]
    for i in 0..coords.len() {
        clamped_coord.push(coords[i].clamp(0, shape[i] - 1));
    }

    clamped_coord
}
