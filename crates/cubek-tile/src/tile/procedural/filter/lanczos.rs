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
        // Zero lobes would leave an empty support and divide by zero in the coefficient below.
        // Checked here rather than in a constructor, which a struct literal can bypass. It fires
        // while the kernel expands, so it surfaces on the client's compilation thread, not at the
        // call site.
        comptime!(assert!(self.lobes > 0, "Lanczos: lobes must be non-zero"));
        let lobes = comptime!(self.lobes as f32);
        let x = self.coordinate.evaluate(coordinates);
        let abs_x = x.abs();
        // With `u = pi * x / lobes` the kernel is `sin(lobes * u) * sin(u) / (lobes * u * u)`,
        // which folds both divisions by `lobes` into one comptime coefficient.
        let u = T::new(core::f32::consts::PI / lobes) * x;
        let numerator = if comptime!(self.lobes == 3) {
            // sin(3u) = sin(u) * (3 - 4 sin^2 u), so the product needs a single sine.
            let s = u.sin();
            let s2 = s * s;
            s2 * fma(T::new(-4.0_f32), s2, T::new(3.0_f32))
        } else {
            (T::new(lobes) * u).sin() * u.sin()
        };
        let denominator = T::new(lobes) * (u * u);
        // `select` evaluates both arms, so the x = 0 arm still divides. Substituting a harmless
        // denominator there keeps that division finite; the outer `select` discards its result.
        let safe_denominator = select(abs_x < T::new(1e-7_f32), T::new(1.0_f32), denominator);
        select(
            abs_x < T::new(1e-7_f32),
            T::new(1.0_f32),
            select(
                abs_x < T::new(lobes),
                numerator / safe_denominator,
                T::new(0.0_f32),
            ),
        )
    }
}
