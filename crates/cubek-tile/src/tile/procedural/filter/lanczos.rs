use cubecl::prelude::*;

use crate::Axis;

use super::super::{AffineCoordinate, Recipe, RecipeCoords, RecipeExpand};

/// Windowed-sinc Lanczos filter over an [`AffineCoordinate`].
pub type LanczosAxis<T> = Lanczos<AffineCoordinate<T>>;

/// Construct a [`LanczosAxis`] recipe filtering along a single coordinate axis.
#[cube]
pub fn lanczos_along<T: Float>(
    #[comptime] axis: Axis,
    offset: T,
    coefficient: T,
    #[comptime] lobes: u8,
) -> LanczosAxis<T> {
    LanczosAxis::<T> {
        coordinate: AffineCoordinate::<T> {
            offset,
            coefficient,
            axis,
        },
        lobes,
    }
}

/// Windowed-sinc Lanczos filter over the value of an inner recipe, `sinc(x) * sinc(x / lobes)`
/// inside the support and zero outside it. `lobes` is the half-width of the support in taps: two
/// gives a four-tap kernel, three a six-tap one.
#[derive(CubeType, Clone)]
pub struct Lanczos<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub lobes: u8,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Lanczos<C> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        // Zero lobes would leave an empty support and divide by zero below. Checked here rather
        // than in a constructor, which a struct literal can bypass. It fires while the kernel
        // expands, so it surfaces on the client's compilation thread, not at the call site.
        comptime!(assert!(self.lobes > 0, "Lanczos: lobes must be non-zero"));
        let x = self.coordinate.evaluate(coordinates);
        let abs_x = x.abs();
        let pi_x = T::new(core::f32::consts::PI) * x;
        let lobes = T::cast_from(self.lobes);
        let denominator = (pi_x * pi_x) / lobes;
        // `select` evaluates both arms, so the x = 0 arm still divides. Substituting a harmless
        // denominator there keeps that division finite; the outer `select` discards its result.
        let safe_denominator = select(abs_x < T::new(1e-7_f32), T::new(1.0_f32), denominator);
        select(
            abs_x < T::new(1e-7_f32),
            T::new(1.0_f32),
            select(
                abs_x < lobes,
                (pi_x.sin() * (pi_x / lobes).sin()) / safe_denominator,
                T::new(0.0_f32),
            ),
        )
    }
}
