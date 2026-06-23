use crate::{
    components::fast_div_mod_value,
    definition::{Resample, ResampleArgs},
};
use cubecl::{prelude::*, std::FastDivmod, std::tensor::layout::CoordsDynI};

/// Computes the absolute coordinate of a cube.
#[cube]
pub fn cube_absolute_coord(
    cube_shape: &Sequence<FastDivmod<usize>>,
    cube_strides: &Sequence<FastDivmod<usize>>,
    cube_pos: usize,
) -> CoordsDynI {
    let mut coords = CoordsDynI::new();

    #[unroll]
    for i in 0..cube_shape.len() {
        let (cube_pos_at_dim, _) = cube_strides[i].div_mod(cube_pos);
        let (_, cube_coord) = cube_shape[i].div_mod(cube_pos_at_dim);

        coords.push(cube_coord as i32);
    }

    coords
}

/// Computes the local coordinate within a tile.
#[cube]
pub fn tile_absolute_coord(
    tile_shape: &Sequence<FastDivmod<usize>>,
    tile_strides: &Sequence<FastDivmod<usize>>,
    cube_coord: &CoordsDynI,
    unit_pos: usize,
    #[comptime] vectorized_axis: usize,
    #[comptime] vector_size: usize,
) -> CoordsDynI {
    let mut coords = CoordsDynI::new();

    #[unroll]
    for i in 0..tile_shape.len() {
        let (unit_pos_at_dim, _) = tile_strides[i].div_mod(unit_pos);
        let (_, coord) = tile_shape[i].div_mod(unit_pos_at_dim);

        let tile_dim_size = fast_div_mod_value(&tile_shape[i]);

        let coord = if i == vectorized_axis {
            ((cube_coord[i] as usize * tile_dim_size + coord) * vector_size) as i32
        } else {
            (cube_coord[i] as usize * tile_dim_size + coord) as i32
        };

        coords.push(coord);
    }

    coords
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

/// Map output coordinate to input coordinate using precomputed anchors.
#[cube]
pub fn map_coord(
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
