use cubecl::prelude::*;

use crate::{Axis, Fold, FoldExpand, floor_div_rem};

use super::{Recipe, RecipeCoords, RecipeExpand};

/// The fractional part a rational coordinate mapping leaves behind, scaled:
/// `coefficient * frac((coord[axis] * numerator_scale + numerator_offset) / divisor)`.
///
/// The one term a resampling filter argument needs that is not affine, and so cannot be composed
/// out of [`AffineCoordinate`](super::AffineCoordinate) and [`Sum`](super::Sum): `frac` is a
/// floor, and the floor of a negative numerator is not the truncating division a kernel emits.
/// The fraction is carried in integers rather than evaluated in floats because the residue is then
/// exact however far out the coordinate runs. `coefficient` folds a sign in, so the subtraction in
/// `x = tap - phase` needs no negation recipe.
///
/// The three terms name the same fraction [`PhysicalAxisMap`](crate::PhysicalAxisMap) does and
/// take its sign discipline: an unsigned scale and divisor, a signed offset. They are plain
/// runtime values rather than comptime ones so that a single path serves a ratio fixed when the
/// kernel is compiled and one fixed at launch, which is what [`Fold`] is for: a constant scale,
/// offset or divisor folds away at expand time exactly as a comptime field would, and a genuinely
/// runtime one stays. Like a [`PhysicalAxisMap`], this cannot run an axis backwards; a flip
/// belongs in the coordinate handed to the recipe, not in the fraction.
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
            .fmul(self.numerator_scale)
            .fcast::<i32>()
            .fadd(self.numerator_offset);
        let (_, residue) = floor_div_rem(numerator, self.divisor.fcast::<i32>());
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
