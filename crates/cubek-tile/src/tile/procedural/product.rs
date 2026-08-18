use cubecl::prelude::*;

use super::{Recipe, RecipeCoords, RecipeExpand};

/// Pointwise product of two recipes: `(A * B)(coords) = A(coords) * B(coords)`. Both factors are
/// evaluated at every coordinate; nothing exploits the factorization when they read orthogonal
/// axes, because a source erases its recipe and so cannot restructure a fill around one.
///
/// `#[cube]` cannot parse a two-parameter generic in a struct literal, so instantiate through an
/// alias: `type AxisProduct<T> = Product<AffineCoordinate<T>, AffineCoordinate<T>>;`.
#[derive(CubeType, Clone)]
pub struct Product<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Product<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) * self.rhs.evaluate(coordinates)
    }
}
