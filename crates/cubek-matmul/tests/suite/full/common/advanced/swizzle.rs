mod no_swizzle {
    use super::*;
    use cubek_matmul::definition::SwizzleModes;
    use cubek_std::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::None,
            rhs: SwizzleMode::None,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

mod b32 {
    use super::*;
    use cubek_matmul::definition::SwizzleModes;
    use cubek_std::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B32,
            rhs: SwizzleMode::B32,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

mod b64 {
    use super::*;
    use cubek_matmul::definition::SwizzleModes;
    use cubek_std::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B64,
            rhs: SwizzleMode::B64,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}

mod b128 {
    use super::*;
    use cubek_matmul::definition::SwizzleModes;
    use cubek_std::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B128,
            rhs: SwizzleMode::B128,
            ..Default::default()
        }
    }

    include!("hypercube.rs");
}
