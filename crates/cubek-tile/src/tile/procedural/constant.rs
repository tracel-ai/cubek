use cubecl::prelude::*;

use super::{Recipe, RecipeAxisDeps, RecipeCoords, RecipeExpand};

/// A procedural field holding one value, which may be a runtime scalar.
#[derive(CubeType, Clone)]
pub struct Constant<T: Numeric> {
    pub value: T,
}

#[cube]
impl<T: Numeric> Recipe<T> for Constant<T> {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        self.value
    }
}

impl<T: Numeric> RecipeAxisDeps for ConstantExpand<T> {
    fn addresses(&self, _scope: &Scope, _axis: crate::Axis) -> bool {
        false
    }
}

/// Constant zero. Unlike [`Constant`] it carries no value at all, so a source built
/// from it holds nothing.
#[derive(CubeType, Clone, Copy, Default)]
pub struct Zeros;

#[cube]
impl<T: Numeric> Recipe<T> for Zeros {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        T::from_int(0)
    }
}

impl RecipeAxisDeps for ZerosExpand {
    fn addresses(&self, _scope: &Scope, _axis: crate::Axis) -> bool {
        false
    }
}

/// Constant one, carrying no value for the same reason as [`Zeros`].
#[derive(CubeType, Clone, Copy, Default)]
pub struct Ones;

#[cube]
impl<T: Numeric> Recipe<T> for Ones {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        T::from_int(1)
    }
}

impl RecipeAxisDeps for OnesExpand {
    fn addresses(&self, _scope: &Scope, _axis: crate::Axis) -> bool {
        false
    }
}
