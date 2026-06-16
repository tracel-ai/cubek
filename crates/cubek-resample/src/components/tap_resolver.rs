use crate::definition::{BoundaryMode, Kernel, Resample, ResampleArgs};
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
        start_coords: &Sequence<i32>,
        args: &ResampleArgs,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] resampling_vectorized_axis: bool,
    ) -> (Vector<F, N>, Vector<F, N>) {
        let input_shape = input.shape();

        if resampling_vectorized_axis {
            resolve_vectorized_tap(
                tap_idx,
                input,
                &input_shape,
                out_coord,
                start_coords,
                args,
                config,
                vectorized_axis,
            )
        } else {
            resolve_scalar_tap(
                tap_idx,
                input,
                &input_shape,
                out_coord,
                start_coords,
                args,
                config,
                vectorized_axis,
            )
        }
    }
}

/// Resolve taps for vectorized and resampling axis.
#[cube]
fn resolve_vectorized_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    input_shape: &CoordsDyn,
    out_coord: &CoordsDyn,
    start_coords: &Sequence<i32>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let mut weight = Vector::empty();
    let mut value = Vector::empty();

    let vector_size = N::value();

    #[unroll]
    for lane in 0..vector_size {
        let (in_coord, clamped_coord) =
            resolve_tap_coords(tap_idx, out_coord, input_shape, start_coords, config, lane);

        let lane_weight = compute_weight::<F, N>(
            out_coord,
            &in_coord,
            &clamped_coord,
            args,
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

/// Resolve taps for non-vectorized or non-resampling axis.
#[cube]
fn resolve_scalar_tap<F: Float, N: Size>(
    tap_idx: usize,
    input: &View<'_, Vector<F, N>, CoordsDyn>,
    input_shape: &CoordsDyn,
    out_coord: &CoordsDyn,
    start_coords: &Sequence<i32>,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let (in_coord, clamped_coord) = resolve_tap_coords(
        tap_idx,
        out_coord,
        input_shape,
        start_coords,
        config,
        0_usize,
    );

    let weight = compute_weight::<F, N>(
        out_coord,
        &in_coord,
        &clamped_coord,
        args,
        config,
        vectorized_axis,
        0_usize,
    );

    let value = input.read(clamped_coord);

    (value, Vector::new(weight))
}

/// Helper to map and clamp coordinates for a specific lane.
#[cube]
fn resolve_tap_coords(
    tap_idx: usize,
    out_coord: &CoordsDyn,
    input_shape: &CoordsDyn,
    start_coords: &Sequence<i32>,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) -> (Sequence<i32>, CoordsDyn) {
    let mut in_coord = from_coords_dyn(out_coord);

    map_coord(tap_idx, &mut in_coord, start_coords, config, lane);

    let clamped_coord = clamp_to_coords_dyn(input_shape, &in_coord);

    (in_coord, clamped_coord)
}

/// Map output coordinate to input coordinate using precomputed anchors.
#[cube]
pub fn map_coord(
    tap_idx: usize,
    in_coord: &mut Sequence<i32>,
    start_coords: &Sequence<i32>,
    #[comptime] config: &Resample,
    #[comptime] lane: usize,
) {
    let mut current_flat_idx = tap_idx;
    let num_axes = comptime!(config.resample_axes.len());

    #[unroll]
    for axis_idx in comptime!(0..num_axes) {
        let resample_axis = config.resample_axes.index(axis_idx);
        let num_taps = Kernel::num_taps(&resample_axis.kernel);

        let tap_1d_idx = current_flat_idx % num_taps;
        current_flat_idx /= num_taps;

        let flat_idx = lane * num_axes + axis_idx;

        in_coord[resample_axis.axis] = start_coords[flat_idx] + tap_1d_idx as i32;
    }
}

/// Convert CoordsDyn to Sequence<i32>.
#[cube]
pub fn from_coords_dyn(coords: &CoordsDyn) -> Sequence<i32> {
    let mut coords_i32 = Sequence::new();

    #[unroll]
    for i in 0..coords.len() {
        coords_i32.push(coords[i] as i32);
    }

    coords_i32
}

/// Clamp coordinates from Sequence<i32> to CoordsDyn, with bounds check.
#[cube]
pub fn clamp_to_coords_dyn(shape: &CoordsDyn, coords: &Sequence<i32>) -> CoordsDyn {
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
    in_coord: &Sequence<i32>,
    clamped_coord: &CoordsDyn,
    args: &ResampleArgs,
    #[comptime] config: &Resample,
    #[comptime] vectorized_axis: usize,
    #[comptime] lane: usize,
) -> F {
    match config.boundary {
        BoundaryMode::Clamp => {
            Kernel::weight::<F, N>(in_coord, out_coord, args, config, vectorized_axis, lane)
        }
        BoundaryMode::Zero => select(
            is_in_bounds(in_coord, &clamped_coord),
            Kernel::weight::<F, N>(in_coord, out_coord, args, config, vectorized_axis, lane),
            F::zero(),
        ),
    }
}

/// Check if coordinate is in bounds depending on boundary mode.
#[cube]
fn is_in_bounds(in_coord: &Sequence<i32>, clamped_coord: &CoordsDyn) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for i in 0..in_coord.len() {
        in_bounds &= in_coord[i] == clamped_coord[i] as i32;
    }

    in_bounds
}
