use cubecl::prelude::*;

use crate::{Axis, Integer, IntegerExpand, floor_div_rem};

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// The fractional part a rational coordinate mapping leaves behind, scaled:
/// `coefficient * frac((coord[axis] * numerator_scale + numerator_offset) / divisor)`.
///
/// The one non-affine term a resampling filter argument needs, so not composable out of
/// [`AffineCoordinate`](super::AffineCoordinate) and [`Sum`](super::Sum): `frac` is a floor, not
/// the truncating division a kernel emits. Integers keep the residue exact however far out it runs.
///
/// The three terms name the same fraction [`PhysicalAxisMap`](crate::PhysicalAxisMap) does, with
/// its sign discipline (unsigned scale and divisor, signed offset). They are runtime values under
/// [`Integer`], so a constant folds away at expand time like a comptime field; a runtime one stays.
///
/// `coefficient` folds a sign in, so `x = tap - phase` needs no negation recipe. Like a
/// [`PhysicalAxisMap`], this cannot run an axis backwards; a flip belongs in the coordinate.
#[derive(CubeType, Clone)]
pub struct Phase<T: Float> {
    /// Multiplies the whole fraction, unlike the two terms below, which sit inside the numerator.
    pub coefficient: T,
    pub numerator_scale: u32,
    pub numerator_offset: i32,
    pub divisor: u32,
    #[cube(comptime)]
    pub axis: Axis,
}

#[cube]
impl<T: Float> Recipe<T> for Phase<T> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let divisor = self.divisor.constant();
        // Zero is the one degenerate divisor an unsigned type still admits. Catchable only when it
        // folds, which is every divisor but a launch-time one, and checked here rather than in a
        // constructor because a struct literal bypasses one.
        comptime!(assert!(
            divisor != Some(0),
            "Phase: divisor must be non-zero"
        ));
        let numerator = coordinates
            .along(self.axis)
            .times(self.numerator_scale)
            .cast::<i32>()
            .plus(self.numerator_offset);
        let (_, residue) = floor_div_rem(numerator, self.divisor.cast::<i32>());
        let residue = T::cast_from(residue);
        // A folded divisor becomes a reciprocal literal to multiply by; only a launch-time one
        // pays for the cast and the divide.
        let fraction = if comptime!(divisor.is_some()) {
            residue * T::new(comptime!(1.0 / divisor.unwrap() as f32))
        } else {
            residue / T::cast_from(self.divisor)
        };
        self.coefficient * fraction
    }
}

impl<T: Float> Reads for PhaseExpand<T> {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        self.axis == axis
    }
}
