use cubecl::{self as cubecl, cube, prelude::*};

/// The factor a row is scaled by, `1 / sqrt(sum / dim + epsilon)` for a row whose squares sum to
/// `sum`: the epsilon a call passes, the dim its kernel is compiled for.
#[derive(CubeType)]
pub struct RmsNormScale {
    epsilon: f32,
    #[cube(comptime)]
    dim: usize,
}

#[cube]
impl RmsNormScale {
    pub fn new(epsilon: f32, #[comptime] dim: usize) -> Self {
        RmsNormScale { epsilon, dim }
    }

    /// The factor of a row whose squares sum to `sum`.
    pub fn of(&self, sum: f32) -> f32 {
        let dim = f32::cast_from(comptime!(self.dim as u32));
        1.0f32 / (sum / dim + self.epsilon).sqrt()
    }
}
