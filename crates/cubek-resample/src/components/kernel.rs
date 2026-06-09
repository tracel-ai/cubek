use crate::definition::Kernel;
use cubecl::prelude::*;

/// The number of taps in the kernel.
#[cube]
pub fn kernel_num_taps(#[comptime] kernel: Kernel) -> usize {
    match kernel {
        Kernel::One => 1,
        Kernel::Uniform { .. } => 1,
        Kernel::Triangle => 2,
        Kernel::Cubic { .. } => 4,
        Kernel::Lanczos { lobes } => 2 * lobes as usize,
    }
}

/// Evaluate the kernel weight where x is the distance from center.
#[cube]
pub fn kernel_weight<C: Float>(#[comptime] kernel: Kernel, x: C) -> C {
    match kernel {
        Kernel::One => C::new(1.0),
        Kernel::Uniform { scale } => C::new(1.0 / scale as f32),
        Kernel::Triangle => {
            let abs_x = C::abs(x);
            let one = C::new(1.0);
            let zero = C::new(0.0);
            if abs_x < one { one - abs_x } else { zero }
        }
        Kernel::Cubic { a } => {
            let a = C::new(a);
            let abs_x = C::abs(x);
            let one = C::new(1.0);
            let two = C::new(2.0);
            let x2 = abs_x * abs_x;
            let x3 = x2 * abs_x;
            if abs_x < one {
                (a + two) * x3 - (a + C::new(3.0)) * x2 + one
            } else if abs_x < two {
                a * x3 - C::new(5.0) * a * x2 + C::new(8.0) * a * abs_x - C::new(4.0) * a
            } else {
                C::new(0.0)
            }
        }
        Kernel::Lanczos { lobes } => {
            let abs_x = C::abs(x);
            let zero = C::new(0.0);
            let eps = C::new(0.001);
            let lobes_f = C::cast_from(lobes);
            if abs_x < eps {
                C::new(1.0)
            } else if abs_x >= lobes_f {
                zero
            } else {
                let pi_val = C::new(3.14159265);
                let pi_x = pi_val * abs_x;
                let pi_x_lobes = pi_x / lobes_f;
                (C::sin(pi_x) / pi_x) * (C::sin(pi_x_lobes) / pi_x_lobes)
            }
        }
    }
}
