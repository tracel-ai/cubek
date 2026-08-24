use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::SeparableRecipeExpand;
use crate::{Axis, Coords, Space};

/// The absolute logical coordinates a [`Recipe`] is evaluated at: the source's `origin` plus the
/// position it is read at, within its [`Space`]. Rebased one axis at a time on demand, so a recipe
/// emits an add only for the axes it actually reads, and one that ignores its coordinates emits none.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RecipeCoords {
    origin: Coords<u32>,
    offset: Coords<u32>,
    #[cube(comptime)]
    pub space: Space,
}

#[cube]
impl RecipeCoords {
    pub(crate) fn new(
        origin: &Coords<u32>,
        offset: &Coords<u32>,
        #[comptime] space: Space,
    ) -> Self {
        RecipeCoords {
            origin: origin.clone(),
            offset: offset.clone(),
            space,
        }
    }

    /// The absolute coordinate along the axis at comptime position `p`.
    pub fn at(&self, #[comptime] p: usize) -> u32 {
        self.origin.at(p) + self.offset.at(p)
    }

    /// The absolute coordinate along the specified `axis`.
    pub fn along(&self, #[comptime] axis: Axis) -> u32 {
        self.at(comptime!(self.space.position(axis)))
    }
}

/// An N-dimensional scalar field evaluated at absolute logical coordinates.
///
/// Recipes implement [`Recipe<T>`] for any numeric element type `T: Numeric` (integers and floats),
/// though continuous interpolation and filtering recipes (such as [`Linear`](super::Linear),
/// [`Cubic`](super::Cubic), [`Lanczos`](super::Lanczos)) are defined over [`Float`] elements.
#[cube(expand_base_traits = "ExpandTypeClone")]
pub trait Recipe<T: Numeric> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T;
}

pub(crate) trait RecipeOps<T: Numeric> {
    fn evaluate_virtual(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> NativeExpand<T>;
}

pub(crate) trait SeparableRecipeOps<T: Numeric>: RecipeOps<T> {
    fn factors_virtual(&self, scope: &Scope) -> usize;
    fn evaluate_factor_virtual(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<T>;
}

impl<T: Numeric, R: SeparableRecipeExpand<T> + RecipeExpand<T>> SeparableRecipeOps<T> for R {
    fn factors_virtual(&self, scope: &Scope) -> usize {
        self.__expand_factors_method(scope)
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
pub struct VirtualRecipe<T: Numeric>(PhantomData<T>);

#[derive(Clone)]
pub struct VirtualRecipeExpand<T: Numeric> {
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
    ) -> VirtualRecipeExpand<T> {
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

    pub fn evaluate_factor(&self, _coordinates: &RecipeCoords, _factor: usize) -> T {
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
