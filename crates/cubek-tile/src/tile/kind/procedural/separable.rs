//! A recipe stated as one factor per contracted axis, which is what lets a contraction
//! evaluate each factor once per tap run instead of the whole product at every point.

use cubecl::prelude::*;

use super::{
    Recipe, RecipeAxisDependencies, RecipeCoords, RecipeExpand, SeparableRecipe,
    SeparableRecipeAxisDependencies, SeparableRecipeExpand,
};

/// The product of one factor per contracted axis, in contraction order: the separable kernel
/// `K₀ ⊗ K₁ ⊗ … ⊗ Kₙ₋₁`. Rank is the sequence's length, so one type serves a 1-D, 2-D or
/// volumetric filter, each factor reading its own axis of the same recipe coordinates.
///
/// Each factor states its own axis, so nothing here checks that they are distinct; a factor
/// reading an axis another one also reads makes the separable evaluation below wrong rather than
/// merely redundant.
#[derive(CubeType, Clone)]
pub struct SeparableProduct<R: CubeType> {
    pub factors: Sequence<R>,
}

impl<R: CubeType> SeparableRecipeAxisDependencies for SeparableProductExpand<R>
where
    R::ExpandType: RecipeAxisDependencies,
{
    fn factor_reads_axis(&self, scope: &Scope, factor: usize, axis: crate::Axis) -> bool {
        let factor = NativeExpand::from_lit(scope, factor);
        self.factors
            .__expand_index_method(scope, factor)
            .reads_axis(scope, axis)
    }
}

/// Construct a [`SeparableProduct`] from its factors, for the reason [`sum_of`](super::sum_of)
/// exists.
#[cube]
pub fn separable_product<R: CubeType>(factors: Sequence<R>) -> SeparableProduct<R> {
    SeparableProduct::<R> { factors }
}

#[cube]
impl<R: CubeType> SeparableProduct<R> {
    /// The sequence's length, refused when empty: both readings below start at factor zero, so an
    /// empty product is caught where the rank is stated rather than at the index that would trip
    /// over it or, worse, in a consumer walking a rank of zero and leaving its accumulator alone.
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
