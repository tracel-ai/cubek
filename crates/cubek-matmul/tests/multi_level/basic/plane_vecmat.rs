//! Inferred-blueprint smoke tests for the plane vec-mat inner-product routines, whose tile matmul
//! takes only a col rhs.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, Product};

#[test]
fn simple_vecmat() {
    let vecmat = ByStrategy::new(
        MultiLevel::SimpleVecMat(Default::default()),
        Declines::AsItStates,
    );
    vecmat.check([Product::of(F16).m(1).n(128).k(128).rhs_along_k()]);
}

#[test]
fn double_vecmat() {
    let vecmat = ByStrategy::new(
        MultiLevel::DoubleVecMat(Default::default()),
        Declines::AsItStates,
    );
    vecmat.check([Product::of(F16).m(1).n(128).k(128).rhs_along_k()]);
}
