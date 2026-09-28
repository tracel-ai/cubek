//! Expansion-time erasure of a recipe, so a tile holds one whatever its type.

use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::{FactorReads, Recipe, RecipeCoords, RecipeCoordsExpand, RecipeExpand, SeparableExpand};
use crate::Axis;

#[doc(hidden)]
pub trait RecipeCall<T: Numeric> {
    fn call(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> NativeExpand<T>;
}

#[doc(hidden)]
pub trait SeparableCall<T: Numeric>: RecipeCall<T> {
    fn call_factors(&self, scope: &Scope) -> usize;
    fn call_factor_reads(&self, scope: &Scope, factor: usize, axis: Axis) -> bool;
    fn call_factor(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T>;
}

impl<T: Numeric, R> SeparableCall<T> for R
where
    R: SeparableExpand<T> + RecipeExpand<T> + FactorReads,
{
    fn call_factors(&self, scope: &Scope) -> usize {
        self.__expand_factors_method(scope)
    }
    fn call_factor_reads(&self, scope: &Scope, factor: usize, axis: Axis) -> bool {
        self.factor_reads(scope, factor, axis)
    }
    fn call_factor(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T> {
        self.__expand_factor_method(scope, coordinates, factor)
    }
}

impl<T: Numeric, R: RecipeExpand<T>> RecipeCall<T> for R {
    fn call(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> NativeExpand<T> {
        self.__expand_evaluate_method(scope, coordinates)
    }
}

/// Expansion-time type erasure for a [`Recipe`]. It is never a GPU virtual call.
#[derive(Clone)]
pub(crate) struct ErasedRecipe<T: Numeric>(PhantomData<T>);

#[derive(Clone)]
pub(crate) struct ErasedRecipeExpand<T: Numeric> {
    state: Arc<dyn RecipeCall<T>>,
    separable: Option<Arc<dyn SeparableCall<T>>>,
}

impl<T: Numeric> ErasedRecipe<T> {
    pub fn __expand_new<R: Recipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> ErasedRecipeExpand<T> {
        ErasedRecipeExpand {
            state: Arc::new(recipe),
            separable: None,
        }
    }

    pub fn __expand_new_separable<R: super::Separable<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> ErasedRecipeExpand<T>
    where
        R::ExpandType: SeparableCall<T>,
    {
        let recipe = Arc::new(recipe);
        ErasedRecipeExpand {
            state: recipe.clone(),
            separable: Some(recipe),
        }
    }

    pub fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        unexpanded!()
    }

    /// The factorization the recipe states, if it states one: one factor per contracted axis.
    /// `None` for a recipe with no separable structure, which is a different answer from a
    /// factorization of rank one and reaches a different contraction schedule.
    pub fn factorization(&self) -> comptime_type!(Option<usize>) {
        unexpanded!()
    }

    pub(crate) fn factor(&self, _coordinates: &RecipeCoords, _factor: usize) -> T {
        unexpanded!()
    }

    /// Whether one separable factor reads `axis` from its recipe coordinates.
    #[allow(dead_code)] // Reached through its expand, from [`Procedural::factor_reads`].
    pub(crate) fn factor_reads(&self, _factor: usize, _axis: Axis) -> comptime_type!(bool) {
        unexpanded!()
    }
}

impl<T: Numeric> ErasedRecipeExpand<T> {
    pub fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
    ) -> NativeExpand<T> {
        self.state.call(scope, coordinates)
    }

    pub fn __expand_factorization_method(&self, scope: &Scope) -> Option<usize> {
        self.separable
            .as_ref()
            .map(|separable| separable.call_factors(scope))
    }

    pub fn __expand_factor_method(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T> {
        match &self.separable {
            Some(separable) => separable.call_factor(scope, coordinates, factor),
            // An unfactorized recipe is still its own factor zero, which keeps this total for a
            // consumer that reached it without asking `factors` first.
            None => {
                assert_eq!(factor, 0, "recipe states no factorization beyond itself");
                self.state.call(scope, coordinates)
            }
        }
    }

    pub fn __expand_factor_reads_method(&self, scope: &Scope, factor: usize, axis: Axis) -> bool {
        match &self.separable {
            Some(separable) => separable.call_factor_reads(scope, factor, axis),
            None => true,
        }
    }
}

impl<T: Numeric> CubeType for ErasedRecipe<T> {
    type ExpandType = ErasedRecipeExpand<T>;
}
impl<T: Numeric> IntoExpand for ErasedRecipeExpand<T> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> ExpandTypeClone for ErasedRecipeExpand<T> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl<T: Numeric> IntoMut for ErasedRecipeExpand<T> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> CubeDebug for ErasedRecipeExpand<T> {}
impl<T: Numeric> AsRefExpand for ErasedRecipeExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: Numeric> AsMutExpand for ErasedRecipeExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
