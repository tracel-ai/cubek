use crate::suite::assert_result;
use crate::suite::test_matmul_strategy;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubecl::{frontend::CubePrimitive, ir::AddressType};
use cubecl::{prelude::TensorBinding, zspace::shape};
use cubek_matmul::launch::Strategy;
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::vecmat_unit_perpendicular::VecMatUnitPerpendicularStrategy;

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

#[test]
pub fn test_unit_perpendicular_very_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 128,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_k_larger_than_n() {
    let case = VecMatTestCase {
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_k_smaller_than_n() {
    let case = VecMatTestCase {
        n: 256,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_lhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_rhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_batched() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_uneven_shape() {
    let case = VecMatTestCase {
        n: 32,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_not_same_vectorization() {
    let case = VecMatTestCase {
        n: 128,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

fn test_vecmat(case: VecMatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let problem = case.to_problem();

    test_matmul_strategy(client, problem, case.strategy);
}
