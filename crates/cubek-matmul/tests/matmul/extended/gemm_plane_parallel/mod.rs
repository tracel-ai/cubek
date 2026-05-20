use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::gemm_plane_parallel::GemmPlaneParallelStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

struct GemmTestCase {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub elems: MatmulGlobalElems,
    pub strategy: Strategy,
}

impl GemmTestCase {
    fn to_problem(&self) -> MatmulProblem {
        // Boilerplate assumes lhs row-major, rhs col-major.
        MatmulProblem::from_parameters(
            self.m,
            self.n,
            self.k,
            shape![self.lhs_batch],
            shape![self.rhs_batch],
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
            MatrixLayout::RowMajor,
            None,
            None,
            self.elems.clone(),
            AddressType::U32,
        )
    }

    pub(crate) fn test(self) {
        let client = TestRuntime::client(&Default::default());
        let problem = self.to_problem();
        test_matmul_strategy(client, problem, self.strategy);
    }
}

mod f16_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("plane_parallel.rs");
}

mod f32_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("plane_parallel.rs");
}
