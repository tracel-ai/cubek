use cubecl::prelude::*;

/// The kernel function, it determines the shape of the kernel.
#[derive(Debug, Clone, Copy, PartialEq, CubeType, Eq, Hash)]
pub enum Kernel {
    /// Weight of one.
    One,
    /// Uniform taps.
    Uniform { scale: u8 },
    /// Triangle, support 2.
    Linear,
    /// Cubic convolution.
    Cubic { a_numerator: u8, a_denominator: u8 },
    /// Sinc-sinc function with `lobes` side-lobes (2 or 3).
    Lanczos { lobes: u8 },
}

impl Kernel {
    /// Number of taps in the kernel.
    pub fn num_taps(&self) -> usize {
        match self {
            Kernel::One => 1,
            Kernel::Uniform { .. } => 1,
            Kernel::Linear => 2,
            Kernel::Cubic { .. } => 4,
            Kernel::Lanczos { lobes } => 2 * *lobes as usize,
        }
    }

    /// Computes the weight of the kernel for a given position.
    pub fn weight<F: Float>(&self, x: F) -> F {
        match self {
            Kernel::One => F::new(1.0),
            Kernel::Uniform { scale } => F::new(1.0 / *scale as f32),
            Kernel::Linear => linear_weight(x),
            Kernel::Cubic {
                a_numerator,
                a_denominator,
            } => {
                let a = F::cast_from(*a_numerator) / F::cast_from(*a_denominator);
                let abs_x = F::abs(x);
                let one = F::new(1.0);
                let two = F::new(2.0);
                let x2 = abs_x * abs_x;
                let x3 = x2 * abs_x;
                if abs_x < one {
                    (a + two) * x3 - (a + F::new(3.0)) * x2 + one
                } else if abs_x < two {
                    a * x3 - F::new(5.0) * a * x2 + F::new(8.0) * a * abs_x - F::new(4.0) * a
                } else {
                    F::new(0.0)
                }
            }
            Kernel::Lanczos { lobes } => {
                let abs_x = F::abs(x);
                let zero = F::new(0.0);
                let eps = F::new(0.001);
                let lobes_f = F::cast_from(*lobes);
                if abs_x < eps {
                    F::new(1.0)
                } else if abs_x >= lobes_f {
                    zero
                } else {
                    let pi_val = F::new(core::f32::consts::PI);
                    let pi_x = pi_val * abs_x;
                    let pi_x_lobes = pi_x / lobes_f;
                    (F::sin(pi_x) / pi_x) * (F::sin(pi_x_lobes) / pi_x_lobes)
                }
            }
        }
    }
}

fn linear_weight<F: Float>(x: F) -> F {
    let abs_x = x.abs();
    select(abs_x < F::new(1.0), F::new(1.0) - abs_x, F::new(0.0))
}
