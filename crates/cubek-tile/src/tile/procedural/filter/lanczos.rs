use cubecl::prelude::*;

use crate::{Coords, Space};

use super::super::{Recipe, RecipeExpand};

/// Windowed-sinc Lanczos filter over the value of an inner recipe, `sinc(x) * sinc(x / lobes)`
/// inside the support and zero outside it.
#[derive(CubeType, Clone)]
pub struct Lanczos<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub lobes: u8,
}

impl<C: CubeType> Lanczos<C> {
    /// Two lobes, a four-tap kernel.
    pub fn lanczos_2(coordinate: C) -> Self {
        Self {
            coordinate,
            lobes: 2,
        }
    }

    /// Three lobes, a six-tap kernel.
    pub fn lanczos_3(coordinate: C) -> Self {
        Self {
            coordinate,
            lobes: 3,
        }
    }
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Lanczos<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        // Zero lobes would leave an empty support and divide by zero below. Checked here rather
        // than in a constructor, which a struct literal can bypass. It fires while the kernel
        // expands, so it surfaces on the client's compilation thread, not at the call site.
        comptime!(assert!(self.lobes > 0, "Lanczos: lobes must be non-zero"));
        let x = self.coordinate.evaluate(coordinates, space);
        let abs_x = x.abs();
        let pi_x = T::new(core::f32::consts::PI) * x;
        let lobes = T::cast_from(self.lobes);
        let denominator = (pi_x * pi_x) / lobes;
        // `select` evaluates both arms, so the singularity at x = 0 is divided away rather than
        // branched around.
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
