use cubecl::prelude::*;

use crate::{Coords, Space};

use super::super::{Recipe, RecipeExpand};

/// Triangle filter over the value of an inner recipe, usually an [`AffineCoordinates`](super::super::AffineCoordinates).
#[derive(CubeType, Clone)]
pub struct Linear<C: CubeType> {
    pub coordinate: C,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Linear<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        let x = self.coordinate.evaluate(coordinates, space).abs();
        select(x < T::new(1.0_f32), T::new(1.0_f32) - x, T::new(0.0_f32))
    }
}
