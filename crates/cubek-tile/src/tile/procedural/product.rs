use cubecl::prelude::*;

use super::{Recipe, RecipeCoords, RecipeExpand};

/// A recipe that factorizes into one factor per contracted axis, `R(coords) = ∏ᵢ Rᵢ(coords)`,
/// factor `i` varying only along the `i`-th contracted axis. The gather microkernel uses this
/// stronger contract to evaluate each factor once per 1-D tap walk instead of evaluating the
/// whole product at every point of their Cartesian product.
///
/// The factor count is the recipe's, not the consumer's: a 1-D, 2-D or N-D filter is the same
/// contract with a different `factors`.
#[cube]
pub trait SeparableRecipe<T: Numeric>: Recipe<T> {
    fn factors(&self) -> comptime_type!(usize);
    /// Evaluate the factor at comptime position `factor`, which indexes the contracted axes in
    /// the order the contraction walks them.
    fn evaluate_factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T;
}

/// Pointwise product of two recipes: `(A * B)(coords) = A(coords) * B(coords)`. Both factors are
/// evaluated at every coordinate; the factorization is not exploited, because nothing here states
/// that the two read orthogonal axes. [`SeparableProduct`] is the one that does.
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

/// The product of one factor per contracted axis, in contraction order: the separable kernel
/// `K₀ ⊗ K₁ ⊗ … ⊗ Kₙ₋₁`. Rank is the sequence's length, so one type serves a 1-D, 2-D or
/// volumetric filter, and each factor is free to read a different axis of the same recipe
/// coordinates.
///
/// Each factor states its own axis, so nothing here checks that they are distinct; a factor
/// reading an axis another one also reads makes the separable evaluation below wrong rather than
/// merely redundant.
#[derive(CubeType, Clone)]
pub struct SeparableProduct<R: CubeType> {
    pub factors: Sequence<R>,
}

/// Construct a [`SeparableProduct`] from its factors, for the reason [`sum_of`](super::sum_of)
/// exists.
#[cube]
pub fn separable_product<R: CubeType>(factors: Sequence<R>) -> SeparableProduct<R> {
    SeparableProduct::<R> { factors }
}

#[cube]
impl<R: CubeType> SeparableProduct<R> {
    /// The sequence's length, refused when empty. An empty product names no axis and contributes
    /// no value, and both readings below start at factor zero, so it is caught where the rank is
    /// stated rather than at the index that would trip over it or, worse, in a consumer that
    /// walks a rank of zero and leaves its accumulator untouched.
    pub(crate) fn rank(&self) -> comptime_type!(usize) {
        let rank = self.factors.len();
        comptime!(assert!(
            rank > 0,
            "SeparableProduct: a separable recipe needs at least one factor"
        ));
        rank
    }
}

#[cube]
impl<T: Numeric, R: Recipe<T>> Recipe<T> for SeparableProduct<R> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let rank = self.rank();
        let mut value = self.factors.index(0usize).evaluate(coordinates);

        #[unroll]
        for f in 1..rank {
            value *= self.factors.index(f).evaluate(coordinates);
        }

        value
    }
}

#[cube]
impl<T: Numeric, R: Recipe<T>> SeparableRecipe<T> for SeparableProduct<R> {
    fn factors(&self) -> comptime_type!(usize) {
        self.rank()
    }

    fn evaluate_factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T {
        self.factors.index(factor).evaluate(coordinates)
    }
}
