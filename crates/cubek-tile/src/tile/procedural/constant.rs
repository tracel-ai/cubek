use cubecl::prelude::*;

use crate::{Coords, Space};

use super::{Recipe, RecipeExpand};

/// A constant procedural field.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Constant<T: Numeric> {
    pub value: T,
}

#[cube]
impl<T: Numeric> Recipe<T> for Constant<T> {
    fn evaluate(&self, _coordinates: &Coords<u32>, #[comptime] _space: Space) -> T {
        self.value
    }
}

/// A zero-cost constant zero procedural field.
#[derive(CubeType, Clone, Copy, Default)]
#[expand(derive(Clone, Copy))]
pub struct Zeros;

#[cube]
impl<T: Numeric> Recipe<T> for Zeros {
    fn evaluate(&self, _coordinates: &Coords<u32>, #[comptime] _space: Space) -> T {
        T::from_int(0)
    }
}

/// A zero-cost constant one procedural field.
#[derive(CubeType, Clone, Copy, Default)]
#[expand(derive(Clone, Copy))]
pub struct Ones;

#[cube]
impl<T: Numeric> Recipe<T> for Ones {
    fn evaluate(&self, _coordinates: &Coords<u32>, #[comptime] _space: Space) -> T {
        T::from_int(1)
    }
}
