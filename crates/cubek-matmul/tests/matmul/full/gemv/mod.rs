use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::gemv_unit_perpendicular::GemvUnitPerpendicularStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

enum GemvKind {
    VecMat,
    MatVec,
}

struct GemvTestCase {
    pub out_dim: usize,
    pub k_dim: usize,
    pub vec_batch: usize,
    pub mat_batch: usize,
    pub mat_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
    pub strategy: Strategy,
    pub kind: GemvKind,
}

impl GemvTestCase {
    fn to_problem(&self) -> MatmulProblem {
        match self.kind {
            GemvKind::VecMat => MatmulProblem::from_parameters(
                1,
                self.out_dim,
                self.k_dim,
                shape![self.vec_batch],
                shape![self.mat_batch],
                MatrixLayout::RowMajor,
                self.mat_layout,
                MatrixLayout::RowMajor,
                None,
                None,
                self.elems.clone(),
                AddressType::U32,
            ),
            GemvKind::MatVec => MatmulProblem::from_parameters(
                self.out_dim,
                1,
                self.k_dim,
                shape![self.mat_batch],
                shape![self.vec_batch],
                self.mat_layout,
                MatrixLayout::RowMajor,
                MatrixLayout::RowMajor,
                None,
                None,
                self.elems.clone(),
                AddressType::U32,
            ),
        }
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

    include!("unit_perpendicular.rs");
}

mod f32_ty {
    use super::*;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("unit_perpendicular.rs");
}
