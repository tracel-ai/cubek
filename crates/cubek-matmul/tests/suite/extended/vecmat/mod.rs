use crate::suite::assert_result;
use crate::suite::test_matmul_strategy;
use cubecl::frontend::CubePrimitive;
use cubecl::ir::AddressType;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubecl::{prelude::TensorBinding, zspace::shape};
use cubek_matmul::launch::Strategy;
use cubek_matmul::routines::BlueprintStrategy;

use crate::suite::layout_to_stride_spec;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_matmul::definition::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_std::InputBinding;
use cubek_std::MatrixLayout;
use cubek_test_utils::{BaseInputSpec, DataKind, Distribution, TestInput};

type TestRuntime = cubecl::TestRuntime;

struct VecMatTestCase {
    pub n: usize,
    pub k: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
    pub strategy: Strategy,
}

impl VecMatTestCase {
    fn to_problem(&self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            1,
            self.n,
            self.k,
            shape![self.lhs_batch],
            shape![self.rhs_batch],
            MatrixLayout::RowMajor,
            self.rhs_layout,
            MatrixLayout::RowMajor,
            None,
            None,
            self.elems.clone(),
            AddressType::U32,
        )
    }
}

fn test_vecmat(case: VecMatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let problem = case.to_problem();

    test_matmul_strategy(client, problem, case.strategy);
}

mod f16_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("plane_parallel.rs");
    include!("unit_perpendicular.rs");
}

mod f32_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("plane_parallel.rs");
    include!("unit_perpendicular.rs");
}
