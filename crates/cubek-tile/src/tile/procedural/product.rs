use cubecl::prelude::*;

use super::{Recipe, RecipeCoords, RecipeExpand};

/// Pointwise product of two recipes: `(A * B)(coords) = A(coords) * B(coords)`. Both factors are
/// evaluated at every coordinate; nothing exploits the factorization when they read orthogonal
/// axes, because a source erases its recipe and so cannot restructure a fill around one.
#[derive(CubeType, Clone)]
pub struct Product<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

/// Construct a [`Product`], for the reason [`sum_of`](super::sum_of) exists.
#[cube]
pub fn product_of<A: CubeType, B: CubeType>(lhs: A, rhs: B) -> Product<A, B> {
    Product::<A, B> { lhs, rhs }
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Product<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) * self.rhs.evaluate(coordinates)
    }
}
