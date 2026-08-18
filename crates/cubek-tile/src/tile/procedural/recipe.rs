use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::{Coords, Space};

/// The absolute logical coordinates a [`Recipe`] is evaluated at: the source's `origin` plus the
/// position it is read at. Rebased one axis at a time on demand, so a recipe emits an add only for
/// the axes it actually reads, and one that ignores its coordinates emits none.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AbsoluteCoords {
    origin: Coords<u32>,
    offset: Coords<u32>,
}

#[cube]
impl AbsoluteCoords {
    pub(crate) fn new(origin: &Coords<u32>, offset: &Coords<u32>) -> Self {
        AbsoluteCoords {
            origin: origin.clone(),
            offset: offset.clone(),
        }
    }

    /// The absolute coordinate along the axis at comptime position `p`.
    pub fn at(&self, #[comptime] p: usize) -> u32 {
        self.origin.at(p) + self.offset.at(p)
    }
}

/// An N-dimensional scalar field evaluated at absolute logical coordinates.
#[cube(expand_base_traits = "ExpandTypeClone")]
pub trait Recipe<T: Numeric> {
    fn evaluate(&self, coordinates: &AbsoluteCoords, #[comptime] space: Space) -> T;
}

pub(crate) trait RecipeOps<T: Numeric> {
    fn evaluate_virtual(
        &self,
        scope: &Scope,
        coordinates: &AbsoluteCoordsExpand,
        space: Space,
    ) -> NativeExpand<T>;
}

impl<T: Numeric, R: RecipeExpand<T>> RecipeOps<T> for R {
    fn evaluate_virtual(
        &self,
        scope: &Scope,
        coordinates: &AbsoluteCoordsExpand,
        space: Space,
    ) -> NativeExpand<T> {
        self.__expand_evaluate_method(scope, coordinates, space)
    }
}

/// Expansion-time type erasure for a [`Recipe`]. It is never a GPU virtual call.
#[derive(Clone)]
pub struct VirtualRecipe<T: Numeric>(PhantomData<T>);

#[derive(Clone)]
pub struct VirtualRecipeExpand<T: Numeric> {
    state: Arc<dyn RecipeOps<T>>,
}

impl<T: Numeric> VirtualRecipe<T> {
    pub fn __expand_new<R: Recipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> VirtualRecipeExpand<T> {
        VirtualRecipeExpand {
            state: Arc::new(recipe),
        }
    }

    pub fn evaluate(&self, _coordinates: &AbsoluteCoords, _space: Space) -> T {
        unexpanded!()
    }
}

impl<T: Numeric> VirtualRecipeExpand<T> {
    pub fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &AbsoluteCoordsExpand,
        space: Space,
    ) -> NativeExpand<T> {
        self.state.evaluate_virtual(scope, coordinates, space)
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
