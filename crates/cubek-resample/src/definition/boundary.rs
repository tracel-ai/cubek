use crate::definition::Resample;
use cubecl::{prelude::*, std::tensor::layout::CoordsDynI};

/// Boundary handling mode for out-of-bounds taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum BoundaryMode {
    /// Out-of-bounds taps contribute zero (skip the tap).
    Zero,
    /// Out-of-bounds coordinates are clamped to the nearest valid input coordinate.
    Clamp,
}

#[cube]
impl BoundaryMode {
    pub fn resolve_weight<F: Float, N: Size>(
        weight: &mut F,
        in_coord: &mut CoordsDynI,
        output_shape: &CoordsDynI,
        #[comptime] config: &Resample,
    ) {
        match config.boundary {
            BoundaryMode::Clamp => {
                clamp_coord(in_coord, output_shape, config);
            }
            BoundaryMode::Zero => {
                *weight = select(
                    in_bounds(&*in_coord, output_shape, config),
                    *weight,
                    F::zero(),
                );
            }
        }
    }
}

/// Clamps the given coordinate to the given output shape.
#[cube]
pub fn clamp_coord(
    coord: &mut CoordsDynI,
    output_shape: &CoordsDynI,
    #[comptime] config: &Resample,
) {
    #[unroll]
    for axis_idx in 0..config.num_axes() {
        let resample_axis = config.resample_axes.index(axis_idx);
        let axis = resample_axis.axis;

        coord[axis] = coord[axis].clamp(0, output_shape[axis] - 1);
    }
}

/// Checks if the given coordinate is in bounds for the given output shape.
#[cube]
pub fn in_bounds(
    coord: &CoordsDynI,
    output_shape: &CoordsDynI,
    #[comptime] config: &Resample,
) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for axis_idx in 0..config.num_axes() {
        let resample_axis = config.resample_axes.index(axis_idx);
        let axis = resample_axis.axis;

        if coord[axis] < 0 || coord[axis] >= output_shape[axis] {
            in_bounds = false;
        }
    }
    in_bounds
}
