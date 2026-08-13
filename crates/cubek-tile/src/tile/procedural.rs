//! A memory-free tile source evaluated from logical coordinates.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{Axis, Coords, Fold, FoldExpand, Region, Space};

/// Built-in recipes for a [`TileKind::Procedural`](crate::TileKind::Procedural) source.
///
/// Recipes are comptime values, not runtime callbacks. [`AxisProduct`](Self::AxisProduct)
/// deliberately preserves separability so a later operand reader can cache one factor per axis.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ProceduralRecipe {
    Zero,
    One,
    Uniform { denominator: usize },
    AxisIndex { axis: Axis },
    AxisProduct(Vec<ProceduralRecipe>),
}

impl ProceduralRecipe {
    pub fn zero() -> Self {
        Self::Zero
    }
    pub fn one() -> Self {
        Self::One
    }
    pub fn uniform(denominator: usize) -> Self {
        assert!(
            denominator > 0,
            "ProceduralRecipe::uniform: denominator must be non-zero"
        );
        Self::Uniform { denominator }
    }
    pub fn axis_index(axis: Axis) -> Self {
        Self::AxisIndex { axis }
    }
    pub fn axis_product(factors: Vec<ProceduralRecipe>) -> Self {
        Self::AxisProduct(factors)
    }
}

/// Runtime state of a procedural source. `origin` tracks regions selected by `Tile::at`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct ProceduralData<T: Numeric> {
    origin: Coords<u32>,
    #[cube(comptime)]
    recipe: ProceduralRecipe,
    #[cube(comptime)]
    _marker: PhantomData<T>,
}

#[cube]
impl<T: Numeric> ProceduralData<T> {
    pub(crate) fn new(#[comptime] space: Space, #[comptime] recipe: ProceduralRecipe) -> Self {
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for _ in 0..comptime!(space.rank()) {
            origin.push(0u32.runtime());
        }
        ProceduralData::<T> {
            origin,
            recipe,
            _marker: PhantomData,
        }
    }

    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> Self {
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            let edge = comptime!(space.partitioner().edge(axis) as u32);
            origin.push(self.origin.at(p) + region.coord(axis).fcast::<u32>() * edge);
        }
        ProceduralData::<T> {
            origin,
            recipe: comptime!(self.recipe.clone()),
            _marker: PhantomData,
        }
    }

    pub(crate) fn evaluate(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        self.value_recipe(pos, space, comptime!(self.recipe.clone()))
    }

    fn value_recipe(
        &self,
        pos: &Coords<u32>,
        #[comptime] space: Space,
        #[comptime] recipe: ProceduralRecipe,
    ) -> T {
        match comptime!(recipe) {
            ProceduralRecipe::Zero => T::from_int(0),
            ProceduralRecipe::One => T::from_int(1),
            ProceduralRecipe::Uniform { denominator } => {
                T::from_int(1) / T::from_int(denominator as i64)
            }
            ProceduralRecipe::AxisIndex { axis } => {
                let p = comptime!(space.position(axis));
                T::cast_from(self.origin.at(p) + pos.at(p))
            }
            ProceduralRecipe::AxisProduct(factors) => {
                let mut value = T::from_int(1);
                #[allow(clippy::needless_range_loop)]
                #[unroll]
                for i in 0..comptime!(factors.len()) {
                    value *= self.value_recipe(
                        pos,
                        comptime!(space.clone()),
                        comptime!(factors[i].clone()),
                    );
                }
                value
            }
        }
    }
}
