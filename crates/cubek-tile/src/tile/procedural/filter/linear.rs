use cubecl::prelude::*;

use crate::Axis;

use super::super::{AffineCoordinate, Recipe, RecipeCoords, RecipeExpand};

/// A [`Linear`] filter applied directly to an [`AffineCoordinate`].
pub type LinearAxis<T> = Linear<AffineCoordinate<T>>;

/// Construct a [`LinearAxis`] recipe filtering along a single coordinate axis.
#[cube]
pub fn linear_along<T: Float>(#[comptime] axis: Axis, offset: T, coefficient: T) -> LinearAxis<T> {
    LinearAxis::<T> {
        coordinate: AffineCoordinate::<T> {
            offset,
            coefficient,
            axis,
        },
    }
}

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
