use cubecl::prelude::*;
use cubecl_common::Ratio;

use crate::Axis;

use super::super::{AffineCoordinate, Recipe, RecipeCoords, RecipeExpand};

/// Keys' cubic-convolution filter over an [`AffineCoordinate`].
pub type CubicAxis<T> = Cubic<AffineCoordinate<T>>;

/// Construct a [`CubicAxis`] recipe filtering along a single coordinate axis.
#[cube]
pub fn cubic_along<T: Float>(
    #[comptime] axis: Axis,
    offset: T,
    coefficient: T,
    #[comptime] a: Ratio,
) -> CubicAxis<T> {
    CubicAxis::<T> {
        coordinate: AffineCoordinate::<T> {
            offset,
            coefficient,
            axis,
        },
        a,
    }
}

/// Keys' cubic-convolution filter over the value of an inner recipe. `a` shapes the kernel:
/// `-1/2` is the interpolating member of the family, `-3/4` the sharper one image resamplers
/// usually pick.
#[derive(CubeType, Clone)]
pub struct Cubic<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub a: Ratio,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Cubic<C> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let a = comptime!(self.a.as_f32());
        let x = self.coordinate.evaluate(coordinates).abs();
        let x2 = x * x;
        let first = fma(
            fma(T::new(comptime!(a + 2.0)), x, T::new(comptime!(-(a + 3.0)))),
            x2,
            T::new(1.0_f32),
        );
        let second = fma(
            fma(
                fma(T::new(comptime!(a)), x, T::new(comptime!(-5.0 * a))),
                x,
                T::new(comptime!(8.0 * a)),
            ),
            x,
            T::new(comptime!(-4.0 * a)),
        );
        select(
            x <= T::new(1.0_f32),
            first,
            select(x <= T::new(2.0_f32), second, T::new(0.0_f32)),
        )
    }
}
