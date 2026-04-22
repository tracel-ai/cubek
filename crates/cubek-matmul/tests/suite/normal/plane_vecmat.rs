//! Inferred-blueprint smoke tests for the plane vec-mat inner-product routines.
//!
//! The tile matmul used here only supports a ColMajor rhs, which is why these
//! tests don't use the default row-row `rect` helper.

use cubecl::{ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
};
use cubek_std::MatrixLayout;

use super::common::{client, f16_elems};
use crate::suite::test_matmul_strategy;

fn vecmat_problem(n: usize, k: usize, elems: MatmulGlobalElems) -> MatmulProblem {
    // (1, n, k) vec-mat shape; rhs must be ColMajor for the inner-product tile.
    MatmulProblem::from_parameters(
        1,
        n,
        k,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    )
}

#[test]
fn simple_vecmat() {
    test_matmul_strategy(
        client(),
        vecmat_problem(128, 128, f16_elems()),
        Strategy::SimpleVecMat(Default::default()),
    );
}

#[test]
fn double_vecmat() {
    test_matmul_strategy(
        client(),
        vecmat_problem(128, 128, f16_elems()),
        Strategy::DoubleVecMat(Default::default()),
    );
}
