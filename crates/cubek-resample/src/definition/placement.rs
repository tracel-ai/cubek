use cubecl::prelude::*;

/// Coordinate map: output index to source coordinate.
#[derive(Debug, Clone, PartialEq, CubeType)]
pub enum Placement {
    /// Continuous affine slide: `start = scale * pos + offset`.
    Continuous { scale: f32, offset: f32 },
    /// Windowed: `start = step * pos − padding`.
    Windowed { step: usize, padding: usize },
}

#[cube]
impl Placement {
    /// Map output position to source coordinate.
    pub fn map<F: Float>(out_pos: usize, #[comptime] placement: &Placement) -> F {
        match placement {
            Placement::Continuous { scale, offset } => {
                F::cast_from(out_pos) * F::cast_from(*scale) + F::cast_from(*offset)
            }

            Placement::Windowed { step, padding } => F::cast_from(out_pos * *step - *padding),
        }
    }
}

impl Eq for Placement {}

// Hash implementation to fix f32 `#[derive(Hash)]` error.
impl core::hash::Hash for Placement {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Placement::Continuous { scale, offset } => {
                scale.to_bits().hash(state);
                offset.to_bits().hash(state);
            }
            Placement::Windowed { step, padding } => {
                step.hash(state);
                padding.hash(state);
            }
        }
    }
}
