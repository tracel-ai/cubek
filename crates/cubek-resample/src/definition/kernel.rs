use crate::definition::{Resample, ResampleAxis};
use cubecl::prelude::*;

/// The kernel function, it determines the shape of the kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub enum Kernel {
    /// A kernel that always returns zero.
    Zero,
    /// Uniform taps with distribution: `1.0 / scale`.
    Uniform { scale: u8 },
    /// Triangle, support 2.
    Linear,
    /// Cubic convolution.
    Cubic { a_numerator: i8, a_denominator: u8 },
    /// Sinc-sinc function with `lobes` side-lobes (2 or 3).
    Lanczos { lobes: u8 },
}

impl Kernel {
    pub fn zero() -> Self {
        Kernel::Zero
    }

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
    /// Compute the combined weight from already-mapped coordinates across all resample axes.
    pub fn weight<F: Float, N: Size>(
        in_coord: &Sequence<i32>,
        centers: &Sequence<F>,
        #[comptime] config: &Resample,
        #[comptime] lane: usize,
    ) -> F {
        let mut weight = F::new(1.0);

        let num_axes = comptime!(config.resample_axes.len());

        for axis_idx in comptime!(0..num_axes) {
            let resample_axis = config.resample_axes.index(axis_idx);

            let lane_idx = lane * num_axes + axis_idx;

            weight *= weight_separable::<F>(lane_idx, in_coord, centers, resample_axis);
        }

        weight
    }
}

/// Computes the weight of a single kernel.
#[cube]
fn weight_separable<F: Float>(
    lane_idx: usize,
    in_coord: &Sequence<i32>,
    centers: &Sequence<F>,
    #[comptime] resample_axis: &ResampleAxis,
) -> F {
    match resample_axis.kernel {
        Kernel::Zero => F::new(0.0),
        Kernel::Uniform { scale } => F::new(1.0) / F::cast_from(scale),
        Kernel::Linear | Kernel::Cubic { .. } | Kernel::Lanczos { .. } => {
            let frac = F::cast_from(in_coord[resample_axis.axis]) - centers[lane_idx];

            match resample_axis.kernel {
                Kernel::Linear => linear_weight::<F>(frac),
                Kernel::Cubic {
                    a_numerator,
                    a_denominator,
                } => cubic_weight::<F>(frac, a_numerator, a_denominator),
                Kernel::Lanczos { lobes } => lanczos_weight::<F>(frac, lobes),
                _ => unreachable!(),
            }
        }
    }
}

/// Computes the linear weight for a given fractional position.
#[cube]
fn linear_weight<F: Float>(frac: F) -> F {
    let abs_frac = frac.abs();
    select(abs_frac < F::new(1.0), F::new(1.0) - abs_frac, F::new(0.0))
}

/// Computes the cubic weight for a given fractional position.
#[cube]
fn cubic_weight<F: Float>(
    frac: F,
    #[comptime] a_numerator: i8,
    #[comptime] a_denominator: u8,
) -> F {
    let a = F::cast_from(a_numerator) / F::cast_from(a_denominator);
    let abs_frac = frac.abs();

    let frac2 = abs_frac * abs_frac;
    let frac3 = frac2 * abs_frac;

    // Convolution 1 (|x| <= 1.0)
    let w1 = (a + F::new(2.0)) * frac3 - (a + F::new(3.0)) * frac2 + F::new(1.0);

    // Convolution 2 (1.0 < |x| <= 2.0)
    let w2 = a * frac3 - F::new(5.0) * a * frac2 + F::new(8.0) * a * abs_frac - F::new(4.0) * a;

    select(
        abs_frac <= F::new(1.0),
        w1,
        select(abs_frac <= F::new(2.0), w2, F::new(0.0)),
    )
}

/// Computes the Lanczos weight for a given fractional position.
#[cube]
fn lanczos_weight<F: Float>(frac: F, #[comptime] lobes: u8) -> F {
    let abs_frac = frac.abs();
    let pi_frac = F::cast_from(core::f32::consts::PI) * frac;
    let denom = (pi_frac * pi_frac) / F::cast_from(lobes);
    let safe_denom = select(abs_frac < F::new(1e-7), F::new(1.0), denom);

    select(
        abs_frac < F::new(1e-7),
        F::new(1.0),
        select(
            abs_frac < F::cast_from(lobes),
            (pi_frac.sin() * (pi_frac / F::cast_from(lobes)).sin()) / safe_denom,
            F::new(0.0),
        ),
    )
}
