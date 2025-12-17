mod g16x8x16 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            16,
            8,
            16,
            vec![1],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

mod g256x256x256 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            256,
            256,
            256,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x100x100 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            100,
            100,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x99x100 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            99,
            100,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g100x100x99 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            100,
            99,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

// line_size_lhs != line_size_rhs
#[cfg(feature = "matmul_tests_alt_shapes")]
mod g23x1x17 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            23,
            1,
            17,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}

#[cfg(feature = "matmul_tests_alt_shapes")]
mod g1x256x256 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            1,
            256,
            256,
            vec![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
        )
    }

    include!("../launch.rs");
}
