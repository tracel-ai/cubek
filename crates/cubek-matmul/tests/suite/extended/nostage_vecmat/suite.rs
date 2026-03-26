use crate::suite::assert_result;
use crate::suite::test_matmul_strategy;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubecl::{frontend::CubePrimitive, ir::AddressType};
use cubecl::{prelude::TensorBinding, zspace::shape};
use cubek_matmul::launch::{Strategy, launch_nostage_vecmat};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::nostage_vecmat::{NoStageVecMatRoutine, NoStageVecMatStrategy};

use crate::suite::layout_to_stride_spec;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_matmul::definition::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_std::InputBinding;
use cubek_std::MatrixLayout;
use cubek_test_utils::{BaseInputSpec, DataKind, Distribution, TestInput};

type TestRuntime = cubecl::TestRuntime;

struct NoStageVecMatTestCase {
    pub n: usize,
    pub k: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
}

impl NoStageVecMatTestCase {
    fn into_problem(self) -> MatmulProblem {
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
            self.elems,
            AddressType::U32,
        )
    }
}

#[test]
pub fn test_very_small_square_rhs_row_major() {
    let case = NoStageVecMatTestCase {
        n: 128,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_k_larger_than_n() {
    let case = NoStageVecMatTestCase {
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_k_smaller_than_n() {
    let case = NoStageVecMatTestCase {
        n: 256,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_small_square_rhs_row_major() {
    let case = NoStageVecMatTestCase {
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_large() {
    let case = NoStageVecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_large_broadcast_lhs() {
    let case = NoStageVecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_large_broadcast_rhs() {
    let case = NoStageVecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_large_broadcast_batched() {
    let case = NoStageVecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_uneven_shape() {
    let case = NoStageVecMatTestCase {
        n: 32,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

#[test]
pub fn test_not_same_vectorization() {
    let case = NoStageVecMatTestCase {
        n: 128,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_nostage_vecmat(case);
}

fn test_nostage_vecmat(case: NoStageVecMatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let problem = case.into_problem();

    test_matmul_strategy(
        client,
        problem,
        Strategy::NoStageVecMat(BlueprintStrategy::Inferred(NoStageVecMatStrategy {
            target_num_planes: 8,
        })),
    );
}
