use cubecl::prelude::*;

use super::{Recipe, RecipeCoords, RecipeExpand, RecipeMeta};

/// Pointwise product of two recipes: `(A * B)(coords) = A(coords) * B(coords)`.
///
/// When `A` and `B` are 1D recipes operating along orthogonal axes, `Product<A, B>` forms a 2D
/// separable kernel.
#[derive(CubeType, Clone)]
pub struct Product<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

impl<A: RecipeMeta + CubeType, B: RecipeMeta + CubeType> RecipeMeta for Product<A, B> {
    const HALO: usize = if A::HALO > B::HALO { A::HALO } else { B::HALO };
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Product<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) * self.rhs.evaluate(coordinates)
    }
}
