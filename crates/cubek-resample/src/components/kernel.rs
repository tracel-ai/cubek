use crate::definition::Kernel;
use cubecl::prelude::*;

/// The number of taps in the kernel.
#[cube]
pub fn kernel_num_taps(#[comptime] kernel: &Kernel) -> usize {
    match kernel {
        Kernel::One => 1,
        Kernel::Uniform { .. } => 1,
        Kernel::Triangle => 2,
        Kernel::Cubic { .. } => 4,
        Kernel::Lanczos { lobes } => 2 * *lobes as usize,
    }
}

/// Evaluate the kernel weight where x is the distance from center.
#[cube]
pub fn kernel_weight<F: Float>(x: F, #[comptime] kernel: &Kernel) -> F {
    match kernel {
        Kernel::One => F::new(1.0),
        Kernel::Uniform { scale } => F::new(1.0 / *scale as f32),
        Kernel::Triangle => {
            let abs_x = F::abs(x);
            let one = F::new(1.0);
            let zero = F::new(0.0);
            if abs_x < one { one - abs_x } else { zero }
        }
        Kernel::Cubic { a } => {
            let a = F::new(*a);
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
                let pi_val = F::new(3.14159265);
                let pi_x = pi_val * abs_x;
                let pi_x_lobes = pi_x / lobes_f;
                (F::sin(pi_x) / pi_x) * (F::sin(pi_x_lobes) / pi_x_lobes)
            }
        }
    }
}
