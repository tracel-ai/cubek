use cubecl::prelude::*;

use crate::Axis;

use super::{Recipe, RecipeCoords, RecipeExpand};

/// A one-dimensional affine coordinate expression, `offset + coefficient * coordinate[axis]`.
/// The axis is compile-time metadata; offset and coefficient can be runtime values.
#[derive(CubeType, Clone)]
pub struct AffineCoordinate<T: Numeric> {
    pub offset: T,
    pub coefficient: T,
    #[cube(comptime)]
    pub axis: Axis,
}

#[cube]
impl<T: Numeric> Recipe<T> for AffineCoordinate<T> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.offset + self.coefficient * T::cast_from(coordinates.along(self.axis))
    }
}
