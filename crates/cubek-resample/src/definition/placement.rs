use cubecl::prelude::*;

/// Coordinate map: output index to source coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub enum Placement {
    /// Continuous affine slide: `start = scale * pos + offset`.
    Continuous,
    /// Windowed: `start = step * pos − padding`.
    Windowed,
}

#[derive(CubeType, CubeLaunch)]
pub struct PlacementArgs {
    // Continuous args
    pub scale: f32,
    pub offset: f32,
    // Windowed args
    pub step: usize,
    pub padding: isize,
}

impl PlacementArgs {
    pub fn identity() -> Self {
        Self::windowed(1, 0)
    }

    pub fn continuous(scale: f32, offset: f32) -> Self {
        Self {
            scale,
            offset,
            step: 0,
            padding: 0,
        }
    }

    pub fn windowed(step: usize, padding: isize) -> Self {
        Self {
            scale: 0.0,
            offset: 0.0,
            step,
            padding,
        }
    }

    pub fn to_launch<R: Runtime>(&self) -> PlacementArgsLaunch<R> {
        PlacementArgsLaunch::new(self.scale, self.offset, self.step, self.padding)
    }
}

#[cube]
impl PlacementArgs {
    pub fn map<F: Float>(&self, pos: usize, #[comptime] placement: &Placement) -> F {
        match placement {
            Placement::Continuous => {
                F::cast_from(pos) * F::cast_from(self.scale) + F::cast_from(self.offset)
            }
            Placement::Windowed => F::cast_from((pos * self.step) as isize - self.padding),
        }
    }
}
