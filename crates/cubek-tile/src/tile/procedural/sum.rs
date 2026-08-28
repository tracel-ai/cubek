use cubecl::prelude::*;

use super::{Recipe, RecipeAxisDependencies, RecipeCoords, RecipeExpand};

/// Pointwise sum of two recipes: `(A + B)(coords) = A(coords) + B(coords)`. Composing
/// [`AffineCoordinate`](super::AffineCoordinate) terms through it is how a recipe reads more than
/// one axis, `c0 + c1 * coord[a1] + c2 * coord[a2]` being a sum of two affine terms.
#[derive(CubeType, Clone)]
pub struct Sum<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

/// Construct a [`Sum`]. `#[cube]` cannot parse a two-parameter generic in a struct literal, which
/// this sidesteps; only the literal is affected, so the type itself still spells out in a turbofish.
#[cube]
pub fn sum_of<A: CubeType, B: CubeType>(lhs: A, rhs: B) -> Sum<A, B> {
    Sum::<A, B> { lhs, rhs }
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Sum<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) + self.rhs.evaluate(coordinates)
    }
}

impl<A: CubeType, B: CubeType> RecipeAxisDependencies for SumExpand<A, B>
where
    A::ExpandType: RecipeAxisDependencies,
    B::ExpandType: RecipeAxisDependencies,
{
    fn reads_axis(&self, scope: &Scope, axis: crate::Axis) -> bool {
        self.lhs.reads_axis(scope, axis) || self.rhs.reads_axis(scope, axis)
    }
}
