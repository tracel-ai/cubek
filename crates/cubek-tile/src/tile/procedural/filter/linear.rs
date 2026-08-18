use cubecl::prelude::*;

use super::super::{Recipe, RecipeCoords, RecipeExpand};

/// Triangle filter over the value of an inner recipe, usually an [`AffineCoordinate`](super::super::AffineCoordinate).
#[derive(CubeType, Clone)]
pub struct Linear<C: CubeType> {
    pub coordinate: C,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Linear<C> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let x = self.coordinate.evaluate(coordinates).abs();
        select(x < T::new(1.0_f32), T::new(1.0_f32) - x, T::new(0.0_f32))
    }
}
