use cubecl::prelude::*;
use cubecl_common::Ratio;

use crate::{Coords, Space};

use super::super::{Recipe, RecipeExpand};

/// Keys' cubic-convolution filter over the value of an inner recipe. `a` shapes the kernel;
/// [`catmull_rom`](Self::catmull_rom) and [`sharp`](Self::sharp) pick the two usual values.
#[derive(CubeType, Clone)]
pub struct Cubic<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub a: Ratio,
}

impl<C: CubeType> Cubic<C> {
    /// The interpolating member of the family, `a = -1/2`.
    pub fn catmull_rom(coordinate: C) -> Self {
        Self {
            coordinate,
            a: Ratio::new(-1, 2),
        }
    }

    /// The sharper `a = -3/4` that image resamplers usually pick.
    pub fn sharp(coordinate: C) -> Self {
        Self {
            coordinate,
            a: Ratio::new(-3, 4),
        }
    }
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Cubic<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        let a = T::new(comptime!(self.a.as_f32()));
        let x = self.coordinate.evaluate(coordinates, space).abs();
        let x2 = x * x;
        let x3 = x2 * x;
        let first = (a + T::new(2.0_f32)) * x3 - (a + T::new(3.0_f32)) * x2 + T::new(1.0_f32);
        let second =
            a * x3 - T::new(5.0_f32) * a * x2 + T::new(8.0_f32) * a * x - T::new(4.0_f32) * a;
        select(
            x <= T::new(1.0_f32),
            first,
            select(x <= T::new(2.0_f32), second, T::new(0.0_f32)),
        )
    }
}
