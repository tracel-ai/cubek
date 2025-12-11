mod g16x8x16 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 16,
            n: 8,
            k: 16,
            lhs_batches: vec![1],
            rhs_batches: vec![1],
            out_batches: vec![1],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

mod g256x256x256 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 256,
            n: 256,
            k: 256,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x100x100 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 100,
            n: 100,
            k: 100,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x99x100 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 100,
            n: 99,
            k: 100,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x100x99 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 100,
            n: 100,
            k: 99,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g23x1x17 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 23,
            n: 1,
            k: 17,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}

#[cfg(feature = "matmul_tests_alt_shapes")]
mod g1x256x256 {
    use super::*;
    use cubek_matmul::components::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();
        MatmulProblem {
            m: 1,
            n: 256,
            k: 256,
            lhs_batches: vec![2],
            rhs_batches: vec![2],
            out_batches: vec![2],
            lhs_strides: vec![],
            rhs_strides: vec![],
            lhs_layout: layouts.0,
            rhs_layout: layouts.1,
        }
    }

    include!("../launch.rs");
}
