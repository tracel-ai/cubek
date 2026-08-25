mod g16x8x16 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            16,
            8,
            16,
            shape![1],
            shape![1],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

mod g256x256x256 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            256,
            256,
            256,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

mod g100x100x100 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            100,
            100,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

// vector_size_lhs != vector_size_rhs
mod g100x99x100 {
    use super::*;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            99,
            100,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

// vector_size_lhs != vector_size_rhs
mod g100x100x99 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            100,
            100,
            99,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

// vector_size_lhs != vector_size_rhs
mod g23x1x17 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            23,
            1,
            17,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

mod g1x256x256 {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            1,
            256,
            256,
            shape![2],
            shape![2],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}

mod batched_vecmat {
    use super::*;
    use cubecl::zspace::shape;
    use cubek_matmul::definition::MatmulProblem;

    fn problem() -> MatmulProblem {
        let layouts = layouts();

        MatmulProblem::from_parameters(
            1,
            10,
            5,
            shape![3],
            shape![1],
            layouts.0,
            layouts.1,
            MatrixLayout::RowMajor,
            None,
            None,
            elems(),
            cubecl::ir::AddressType::default(),
        )
    }

    include!("../launch.rs");
}
