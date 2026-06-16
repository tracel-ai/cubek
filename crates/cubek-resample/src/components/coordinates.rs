use crate::definition::{Kernel, Resample};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

pub type CoordsDynI = Sequence<i32>;

/// Helper to map and clamp coordinates for a specific lane.
#[cube]
pub fn resolve_tap_coords(
    tap_idx: usize,
    out_coord: &CoordsDyn,
    input_shape: &CoordsDyn,
    start_coords: &CoordsDynI,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) -> (CoordsDynI, CoordsDyn) {
    let mut in_coord = from_coords_dyn(out_coord);

    map_coord(tap_idx, &mut in_coord, start_coords, config, lane);

    let clamped_coord = clamp_to_coords_dyn(input_shape, &in_coord);

    (in_coord, clamped_coord)
}

/// Map output coordinate to input coordinate using precomputed anchors.
#[cube]
fn map_coord(
    tap_idx: usize,
    in_coord: &mut CoordsDynI,
    start_coords: &CoordsDynI,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) {
    let mut current_flat_idx = tap_idx;
    let num_axes = comptime!(config.resample_axes.len());

    #[unroll]
    for axis_idx in comptime!(0..num_axes) {
        let resample_axis = config.resample_axes.index(axis_idx);
        let num_taps = Kernel::num_taps(&resample_axis.kernel);

        let tap_axis_idx = current_flat_idx % num_taps;
        current_flat_idx /= num_taps;

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
