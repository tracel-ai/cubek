use crate::definition::{Placement, Resample};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

/// The kernel function, it determines the shape of the kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub enum Kernel {
    /// Uniform taps.
    Uniform { scale: u8 },
    /// Triangle, support 2.
    Linear,
    /// Cubic convolution.
    Cubic { a_numerator: i8, a_denominator: u8 },
    /// Sinc-sinc function with `lobes` side-lobes (2 or 3).
    Lanczos { lobes: u8 },
}

impl Kernel {
    pub fn one() -> Self {
        Kernel::Uniform { scale: 1 }
    }

    pub fn cubic_catmull_rom() -> Self {
        Kernel::Cubic {
            a_numerator: -1,
            a_denominator: 2,
        }
    }

    pub fn cubic_sharp() -> Self {
        Kernel::Cubic {
            a_numerator: -3,
            a_denominator: 4,
        }
    }

    pub fn lanczos_2() -> Self {
        Kernel::Lanczos { lobes: 2 }
    }

    pub fn lanczos_3() -> Self {
        Kernel::Lanczos { lobes: 3 }
    }
}

#[cube]
impl Kernel {
    /// Number of taps in the kernel.
    pub fn num_taps(#[comptime] this: &Self) -> usize {
        match this {
            Kernel::Uniform { .. } => 1,
            Kernel::Linear => 2,
            Kernel::Cubic { .. } => 4,
            Kernel::Lanczos { lobes } => 2 * *lobes as usize,
        }
    }

    /// Compute the combined weight from already-mapped coordinates across all resample axes.
    pub fn weight<F: Float>(
        in_coord: &mut CoordsDyn,
        out_coord: &CoordsDyn,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] num_axes: usize,
        #[comptime] lane: usize,
    ) -> F {
        let mut weight = F::new(1.0);

        #[unroll]
        for axis_idx in 0..num_axes {
            let resample_axis = config.resample_axes.index(axis_idx);

            let out_pos = out_coord[resample_axis.axis] as usize;

            let lane_out_pos = if resample_axis.axis == vectorized_axis {
                out_pos + lane
            } else {
                out_pos
            };

            let center = Placement::map::<F>(lane_out_pos, &resample_axis.placement);
            let x = F::cast_from(in_coord[resample_axis.axis]) - center - F::new(1.0);

            weight *= weight_1d::<F>(x, &resample_axis.kernel);
        }

        weight
    }
}

/// Computes the weight of a single kernel for a given fractional position.
#[cube]
fn weight_1d<F: Float>(x: F, #[comptime] kernel: &Kernel) -> F {
    match kernel {
        Kernel::Uniform { scale } => F::new(1.0) / F::cast_from(*scale),
        Kernel::Linear => linear_weight::<F>(x),
        Kernel::Cubic {
            a_numerator,
            a_denominator,
        } => cubic_weight::<F>(x, *a_numerator, *a_denominator),
        Kernel::Lanczos { lobes } => lanczos_weight::<F>(x, *lobes),
    }
}

/// Computes the linear weight for a given fractional position.
#[cube]
fn linear_weight<F: Float>(x: F) -> F {
    let abs_x = x.abs();
    select(abs_x < F::new(1.0), F::new(1.0) - abs_x, F::new(0.0))
}

/// Computes the cubic weight for a given fractional position.
#[cube]
fn cubic_weight<F: Float>(x: F, #[comptime] a_numerator: i8, #[comptime] a_denominator: u8) -> F {
    let a = F::cast_from(a_numerator) / F::cast_from(a_denominator);
    let abs_x = x.abs();

    let x2 = abs_x * abs_x;
    let x3 = x2 * abs_x;

    // Convolution 1 (|x| <= 1.0)
    let w1 = (a + F::new(2.0)) * x3 - (a + F::new(3.0)) * x2 + F::new(1.0);

    // Convolution 2 (1.0 < |x| <= 2.0)
    let w2 = a * x3 - F::new(5.0) * a * x2 + F::new(8.0) * a * abs_x - F::new(4.0) * a;

    select(
        abs_x <= F::new(1.0),
        w1,
        select(abs_x <= F::new(2.0), w2, F::new(0.0)),
    )
}

/// Computes the Lanczos weight for a given fractional position.
#[cube]
fn lanczos_weight<F: Float>(x: F, #[comptime] lobes: u8) -> F {
    let abs_x = x.abs();
    let pi_x = F::cast_from(core::f32::consts::PI) * x;
    let denom = (pi_x * pi_x) / F::cast_from(lobes);
    let safe_denom = select(abs_x < F::new(1e-7), F::new(1.0), denom);

    select(
        abs_x < F::new(1e-7),
        F::new(1.0),
        select(
            abs_x < F::cast_from(lobes),
            (pi_x.sin() * (pi_x / F::cast_from(lobes)).sin()) / safe_denom,
            F::new(0.0),
        ),
    )
}
