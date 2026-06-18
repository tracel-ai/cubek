use crate::definition::CoordsDynI;
use cubecl::prelude::*;

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
        weight: F,
        in_coord: &CoordsDynI,
        clamped_coord: &CoordsDynI,
        #[comptime] this: &Self,
    ) -> F {
        match this {
            BoundaryMode::Clamp => weight,
            BoundaryMode::Zero => select(is_in_bounds(in_coord, clamped_coord), weight, F::zero()),
        }
    }
}

/// Check if coordinate is in bounds depending on boundary mode.
#[cube]
fn is_in_bounds(in_coord: &CoordsDynI, clamped_coord: &CoordsDynI) -> bool {
    let mut in_bounds = true;

    #[unroll]
    for i in 0..in_coord.len() {
        in_bounds &= in_coord[i] == clamped_coord[i];
    }

    in_bounds
}
