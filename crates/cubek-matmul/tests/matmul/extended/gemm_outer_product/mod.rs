use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::gemm_outer_product::GemmOuterProductStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

/// Unified harness for the outer-product matmul: full GEMM (4 layout
/// combinations), vec-mat (m = 1), and mat-vec (n = 1) all run through
/// the same case struct and `Strategy::GemmOuterProduct`.
struct GemmTestCase {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
    pub strategy: Strategy,
}

impl GemmTestCase {
    fn to_problem(&self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            self.m,
            self.n,
            self.k,
            shape![self.lhs_batch],
            shape![self.rhs_batch],
            self.lhs_layout,
            self.rhs_layout,
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

fn outer_product() -> Strategy {
    Strategy::GemmOuterProduct(BlueprintStrategy::Inferred(GemmOuterProductStrategy {
        target_num_planes: None,
    }))
}

mod f16_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("outer_product.rs");
}

mod f32_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("outer_product.rs");
}
