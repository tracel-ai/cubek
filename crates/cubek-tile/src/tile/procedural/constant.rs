use cubecl::prelude::*;

use super::{Recipe, RecipeCoords, RecipeExpand, RecipeMeta};

/// A procedural field holding one value, which may be a runtime scalar.
#[derive(CubeType, Clone)]
pub struct Constant<T: Numeric> {
    pub value: T,
}

impl<T: Numeric> RecipeMeta for Constant<T> {}

#[cube]
impl<T: Numeric> Recipe<T> for Constant<T> {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        self.value
    }
}

/// Constant zero. Unlike [`Constant`] it carries no value at all, so a source built
/// from it holds nothing.
#[derive(CubeType, Clone, Copy, Default)]
pub struct Zeros;

impl RecipeMeta for Zeros {}

#[cube]
impl<T: Numeric> Recipe<T> for Zeros {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        T::from_int(0)
    }
}

/// Constant one, carrying no value for the same reason as [`Zeros`].
#[derive(CubeType, Clone, Copy, Default)]
pub struct Ones;

impl RecipeMeta for Ones {}

#[cube]
impl<T: Numeric> Recipe<T> for Ones {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        T::from_int(1)
    }
}
