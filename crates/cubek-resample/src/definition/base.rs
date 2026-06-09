/// Resampling operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Resample {
    pub kernel: Kernel,
    pub placement: Placement,
    pub semiring: Semiring,
}

/// The semiring, it determines how the values are combined.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Semiring {
    /// Linear algebra: `y = A·x`.
    Linear,
    /// Tropical algebra: `f(x, w) = x + w`.
    Tropical,
    /// Log-sum-exp algebra: `f(x, w) = log(exp(x) + exp(w))`.
    Log,
}

/// The kernel function, it determines the shape of the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Weight of one.
    One,
    /// Uniform taps.
    Uniform { scale: u32 },
    /// Triangle, support 2.
    Triangle,
    /// Cubic convolution.
    Cubic { a: f32 },
    /// Sinc-sinc function with `lobes` side-lobes (2 or 3).
    Lanczos { lobes: u32 },
}

impl Eq for Kernel {}

// Hash implementation to fix f32 `#[derive(Hash)]` error.
impl core::hash::Hash for Kernel {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Kernel::Uniform { scale } => scale.hash(state),
            Kernel::Cubic { a } => a.to_bits().hash(state),
            Kernel::Lanczos { lobes } => lobes.hash(state),
            _ => {}
        }
    }
}

/// Coordinate map: output index to source coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    /// Continuous affine slide: `start = scale * pos + offset`.
    Continuous { scale: f32, offset: f32 },
    /// Windowed: `start = step * pos − pad`.
    Windowed { step: usize, pad: usize },
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
            Placement::Windowed { step, pad } => {
                step.hash(state);
                pad.hash(state);
            }
        }
    }
}
