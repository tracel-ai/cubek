use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{routines::BlueprintStrategy, strategy::Strategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::gemm::GemmStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

/// Unified harness for the gemm family. Drives full GEMM (all 4 layout
/// combinations), vec-mat (m = 1), and mat-vec (n = 1) through the same
/// case struct and `Strategy::Gemm`. The `plane_parallel.rs` cases cover
/// the Row-Col (Dot) variant — the same set runs on any backend; the
/// `outer_product.rs` cases additionally cover Row-Row / Col-Row / Col-Col,
/// which are CPU-only (the family enforces `plane_dim == 1` for those).
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

fn gemm() -> Strategy {
    Strategy::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
        target_num_planes: None,
    }))
}

// Legacy strategy-helper aliases — the test bodies were authored against
// per-routine helpers (`plane_parallel()`, `outer_product()`); both now
// resolve to the unified `gemm()` strategy.
use gemm as plane_parallel;
use gemm as outer_product;

mod f16_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    mod plane_parallel_cases {
        use super::*;
        include!("plane_parallel.rs");
    }

    mod outer_product_cases {
        use super::*;
        include!("outer_product.rs");
    }
}

mod f32_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    mod plane_parallel_cases {
        use super::*;
        include!("plane_parallel.rs");
    }

    mod outer_product_cases {
        use super::*;
        include!("outer_product.rs");
    }
}
