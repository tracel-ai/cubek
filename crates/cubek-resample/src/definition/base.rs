use cubecl::prelude::Sequence;

/// Resampling operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Resample {
    pub axes: Sequence<usize>,
    pub kernel: Kernel,
    pub placement: Placement,
    pub semiring: Semiring,
}

impl Resample {
    pub fn new() -> Self {
        Self {
            axes: Sequence::new(),
            kernel: Kernel::One,
            semiring: Semiring::Linear,
            placement: Placement::Windowed { step: 1, pad: 0 },
        }
    }

    pub fn with_axis(mut self, axis: usize) -> Self {
        self.axes.push(axis);
        self
    }

    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn with_placement(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }

    pub fn with_semiring(mut self, semiring: Semiring) -> Self {
        self.semiring = semiring;
        self
    }
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
