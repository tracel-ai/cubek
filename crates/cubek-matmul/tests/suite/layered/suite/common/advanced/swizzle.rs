#[cfg(not(feature = "matmul_tests_swizzle"))]
mod no_swizzle {
    use super::*;
    use cubek_matmul::components::{SwizzleConfig, stage::SwizzleMode};

    fn swizzle() -> SwizzleConfig {
        SwizzleConfig {
            lhs: SwizzleMode::None,
            rhs: SwizzleMode::None,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

#[cfg(feature = "matmul_tests_swizzle")]
mod b32 {
    use super::*;
    use cubek_matmul::components::{SwizzleConfig, stage::SwizzleMode};

    fn swizzle() -> SwizzleConfig {
        SwizzleConfig {
            lhs: SwizzleMode::B32,
            rhs: SwizzleMode::B32,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

#[cfg(feature = "matmul_tests_swizzle")]
mod b64 {
    use super::*;
    use cubek_matmul::components::{SwizzleConfig, stage::SwizzleMode};

    fn swizzle() -> SwizzleConfig {
        SwizzleConfig {
            lhs: SwizzleMode::B64,
            rhs: SwizzleMode::B64,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

#[cfg(feature = "matmul_tests_swizzle")]
mod b128 {
    use super::*;
    use cubek_matmul::components::{SwizzleConfig, stage::SwizzleMode};

    fn swizzle() -> SwizzleConfig {
        SwizzleConfig {
            lhs: SwizzleMode::B128,
            rhs: SwizzleMode::B128,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}
