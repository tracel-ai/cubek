use cubecl::prelude::*;

use crate::{Axis, Space};

use super::{AbsoluteCoords, Recipe, RecipeExpand};

/// A one-dimensional affine coordinate expression, `offset + coefficient * coordinate[axis]`.
/// The axis is compile-time metadata; offset and coefficient can be runtime values.
#[derive(CubeType, Clone)]
pub struct AffineCoordinates<T: Numeric> {
    pub offset: T,
    pub coefficient: T,
    #[cube(comptime)]
    pub axis: Axis,
}

#[cube]
impl<T: Numeric> Recipe<T> for AffineCoordinates<T> {
    fn evaluate(&self, coordinates: &AbsoluteCoords, #[comptime] space: Space) -> T {
        self.offset
            + self.coefficient * T::cast_from(coordinates.at(comptime!(space.position(self.axis))))
    }
}
