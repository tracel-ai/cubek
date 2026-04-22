use cubecl::{Runtime, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
};
use cubek_std::MatrixLayout;

use crate::suite::test_matmul_strategy;

type TestRuntime = cubecl::TestRuntime;

struct MatmulTestCase {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
}

impl MatmulTestCase {
    fn into_problem(self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            self.m,
            self.n,
            self.k,
            shape![self.batch],
            shape![self.batch],
            self.lhs_layout,
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
pub fn test_very_small() {
    test_naive(MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_very_small_col_major() {
    test_naive(MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_small() {
    test_naive(MatmulTestCase {
        m: 64,
        n: 64,
        k: 64,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_odd() {
    test_naive(MatmulTestCase {
        m: 1,
        n: 255,
        k: 101,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_large() {
    test_naive(MatmulTestCase {
        m: 256,
        n: 256,
        k: 256,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_with_check_bounds() {
    test_naive(MatmulTestCase {
        m: 60,
        n: 60,
        k: 60,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

#[test]
pub fn test_with_batches() {
    test_naive(MatmulTestCase {
        m: 64,
        n: 64,
        k: 64,
        batch: 3,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    });
}

fn test_naive(case: MatmulTestCase) {
    let client = TestRuntime::client(&Default::default());
    let problem = case.into_problem();
    test_matmul_strategy(client, problem, Strategy::Naive);
}
