//! A recipe stated as one factor per contracted axis, which is what lets a contraction
//! evaluate each factor once per tap run instead of the whole product at every point.

use cubecl::prelude::*;

use super::{FactorReads, Reads, Recipe, RecipeCoords, RecipeExpand, Separable, SeparableExpand};

/// The product of one factor per contracted axis, in contraction order: the separable kernel
/// `K₀ ⊗ K₁ ⊗ … ⊗ Kₙ₋₁`. Rank is the sequence's length, so one type serves a 1-D, 2-D or
/// volumetric filter, each factor reading its own axis of the same recipe coordinates.
///
/// Each factor states its own axis, so nothing here checks that they are distinct; a factor
/// reading an axis another one also reads makes the separable evaluation below wrong rather than
/// merely redundant.
#[derive(CubeType, Clone)]
pub struct Factors<R: CubeType> {
    pub factors: Sequence<R>,
}

impl<R: CubeType> FactorReads for FactorsExpand<R>
where
    R::ExpandType: Reads,
{
    fn factor_reads(&self, scope: &Scope, factor: usize, axis: crate::Axis) -> bool {
        let factor = NativeExpand::from_lit(scope, factor);
        self.factors
            .__expand_index_method(scope, factor)
            .reads(scope, axis)
    }
}

#[cube]
impl<R: CubeType> Factors<R> {
    /// The product of `factors`, in contraction order.
    pub fn new(factors: Sequence<R>) -> Factors<R> {
        Factors::<R> { factors }
    }

    /// The sequence's length, refused when empty: both readings below start at factor zero, so an
    /// empty product is caught where the rank is stated rather than at the index that would trip
    /// over it or, worse, in a consumer walking a rank of zero and leaving its accumulator alone.
    pub(crate) fn rank(&self) -> comptime_type!(usize) {
        let rank = self.factors.len();
        comptime!(assert!(
            rank > 0,
            "Factors: a separable recipe needs at least one factor"
        ));
        rank
    }
}

#[cube]
impl<T: Numeric, R: Recipe<T>> Recipe<T> for Factors<R> {
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
impl<T: Numeric, R: Recipe<T>> Separable<T> for Factors<R> {
    fn factors(&self) -> comptime_type!(usize) {
        self.rank()
    }

    fn factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T {
        self.factors.index(factor).evaluate(coordinates)
    }
}
