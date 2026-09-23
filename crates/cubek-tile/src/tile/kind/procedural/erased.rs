//! Expansion-time erasure of a recipe, so a tile holds one whatever its type.

use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::{
    Recipe, RecipeCoords, RecipeCoordsExpand, RecipeExpand, SeparableRecipeAxisDependencies,
    SeparableRecipeExpand,
};
use crate::Axis;

#[doc(hidden)]
pub trait RecipeOps<T: Numeric> {
    fn evaluate_virtual(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> NativeExpand<T>;
}

#[doc(hidden)]
pub trait SeparableRecipeOps<T: Numeric>: RecipeOps<T> {
    fn factors_virtual(&self, scope: &Scope) -> usize;
    fn factor_reads_axis_virtual(&self, scope: &Scope, factor: usize, axis: Axis) -> bool;
    fn evaluate_factor_virtual(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T>;
}

impl<T: Numeric, R> SeparableRecipeOps<T> for R
where
    R: SeparableRecipeExpand<T> + RecipeExpand<T> + SeparableRecipeAxisDependencies,
{
    fn factors_virtual(&self, scope: &Scope) -> usize {
        self.__expand_factors_method(scope)
    }
    fn factor_reads_axis_virtual(&self, scope: &Scope, factor: usize, axis: Axis) -> bool {
        self.factor_reads_axis(scope, factor, axis)
    }
    fn evaluate_factor_virtual(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T> {
        self.__expand_evaluate_factor_method(scope, coordinates, factor)
    }
}

impl<T: Numeric, R: RecipeExpand<T>> RecipeOps<T> for R {
    fn evaluate_virtual(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> NativeExpand<T> {
        self.__expand_evaluate_method(scope, coordinates)
    }
}

/// Expansion-time type erasure for a [`Recipe`]. It is never a GPU virtual call.
#[derive(Clone)]
pub(crate) struct VirtualRecipe<T: Numeric>(PhantomData<T>);

#[derive(Clone)]
pub(crate) struct VirtualRecipeExpand<T: Numeric> {
    state: Arc<dyn RecipeOps<T>>,
    separable: Option<Arc<dyn SeparableRecipeOps<T>>>,
}

impl<T: Numeric> VirtualRecipe<T> {
    pub fn __expand_new<R: Recipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> VirtualRecipeExpand<T> {
        VirtualRecipeExpand {
            state: Arc::new(recipe),
            separable: None,
        }
    }

    pub fn __expand_new_separable<R: super::SeparableRecipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> VirtualRecipeExpand<T>
    where
        R::ExpandType: SeparableRecipeOps<T>,
    {
        let recipe = Arc::new(recipe);
        VirtualRecipeExpand {
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
    pub fn factors(&self) -> comptime_type!(Option<usize>) {
        unexpanded!()
    }

    pub(crate) fn evaluate_factor(&self, _coordinates: &RecipeCoords, _factor: usize) -> T {
        unexpanded!()
    }

    /// Whether one separable factor reads `axis` from its recipe coordinates.
    #[allow(dead_code)] // Reached through its expand, from [`ProceduralData::factor_reads_axis`].
    pub(crate) fn factor_reads_axis(&self, _factor: usize, _axis: Axis) -> comptime_type!(bool) {
        unexpanded!()
    }
}

impl<T: Numeric> VirtualRecipeExpand<T> {
    pub fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
    ) -> NativeExpand<T> {
        self.state.evaluate_virtual(scope, coordinates)
    }

    pub fn __expand_factors_method(&self, scope: &Scope) -> Option<usize> {
        self.separable
            .as_ref()
            .map(|separable| separable.factors_virtual(scope))
    }

    pub fn __expand_evaluate_factor_method(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T> {
        match &self.separable {
            Some(separable) => separable.evaluate_factor_virtual(scope, coordinates, factor),
            // An unfactorized recipe is still its own factor zero, which keeps this total for a
            // consumer that reached it without asking `factors` first.
            None => {
                assert_eq!(factor, 0, "recipe states no factorization beyond itself");
                self.state.evaluate_virtual(scope, coordinates)
            }
        }
    }

    pub fn __expand_factor_reads_axis_method(
        &self,
        scope: &Scope,
        factor: usize,
        axis: Axis,
    ) -> bool {
        match &self.separable {
            Some(separable) => separable.factor_reads_axis_virtual(scope, factor, axis),
            None => true,
        }
    }
}

impl<T: Numeric> CubeType for VirtualRecipe<T> {
    type ExpandType = VirtualRecipeExpand<T>;
}
impl<T: Numeric> IntoExpand for VirtualRecipeExpand<T> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> ExpandTypeClone for VirtualRecipeExpand<T> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl<T: Numeric> IntoMut for VirtualRecipeExpand<T> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> CubeDebug for VirtualRecipeExpand<T> {}
impl<T: Numeric> AsRefExpand for VirtualRecipeExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: Numeric> AsMutExpand for VirtualRecipeExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
