use cubecl::prelude::*;

use cubecl::tile::{
    Dimension,
    procedural::{AffineCoordinate, Reads, Recipe, RecipeCoords, RecipeExpand},
};

/// A [`Linear`] filter applied directly to an [`AffineCoordinate`].
pub type LinearDimension<T> = Linear<AffineCoordinate<T>>;

/// Construct a [`LinearDimension`] recipe filtering along a single coordinate dimension.
#[cube]
pub fn linear_along<T: Float>(
    #[comptime] dimension: Dimension,
    offset: T,
    coefficient: T,
) -> LinearDimension<T> {
    LinearDimension::<T> {
        coordinate: AffineCoordinate::<T> {
            offset,
            coefficient,
            dimension,
        },
    }
}

/// Triangle filter over the value of an inner recipe, usually an [`AffineCoordinate`].
#[derive(CubeType, Clone)]
pub struct Linear<C: CubeType> {
    pub coordinate: C,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Linear<C> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let x = self.coordinate.evaluate(coordinates).abs();
        select(x < T::new(1.0), T::new(1.0) - x, T::new(0.0))
    }
}

impl<C: CubeType> Reads for LinearExpand<C>
where
    C::ExpandType: Reads,
{
    fn reads(&self, scope: &Scope, dimension: Dimension) -> bool {
        self.coordinate.reads(scope, dimension)
    }
}
