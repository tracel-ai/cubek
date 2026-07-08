use crate::matmul::launcher_strategy::run_with_strides;
use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{routines::BlueprintStrategy, strategy::Strategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::gemm::GemmStrategy,
};
use cubek_std::MatrixLayout;
use cubek_test_utils::{TestOutcome, ValidationResult};

type TestRuntime = cubecl::TestRuntime;

/// Unified harness for the gemm family. Drives full GEMM (all 4 layout
/// combinations), vec-mat (m = 1), and mat-vec (n = 1) through the same case
/// struct and `Strategy::Gemm`. `plane_parallel.rs` covers the Row-Col (Dot)
/// variant. `outer_product.rs` covers Row-Row, Col-Row, and Col-Col, which run
/// on CPU directly and on GPU after `launch_ref` normalizes them to Dot.
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

    /// Like [`test`], but asserts the matmul actually ran and validated rather
    /// than letting the test policy accept a config rejection. Use for layouts
    /// that must execute on GPU, where the default `correct` policy would
    /// otherwise silently accept a `CompileError` and hide a regression.
    pub(crate) fn test_executes(self) {
        let client = TestRuntime::client(&Default::default());
        let problem = self.to_problem();
        let outcome = run_with_strides(client, problem, self.strategy);
        assert!(
            matches!(outcome, TestOutcome::Validated(ValidationResult::Pass)),
            "expected the matmul to execute and validate, got {outcome:?}"
        );
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
