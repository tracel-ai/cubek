//! Inferred-blueprint smoke tests for the GEMV routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, Product};

/// Gemm runs a vec-mat against a col rhs on a GPU through its dot variant.
#[cfg(feature = "heavy")]
#[test]
fn gemm_vecmat() {
    let gemm = ByStrategy::new(MultiLevel::Gemm(Default::default()), Declines::AsItStates);
    gemm.check([Product::of(F16).m(1).n(128).k(128).rhs_along_k()]);
}

/// GemvUnitPerpendicular takes only vec-mat products, a single row.
#[test]
fn gemv_unit_perpendicular_vecmat() {
    let gemv = ByStrategy::new(
        MultiLevel::GemvUnitPerpendicular(Default::default()),
        Declines::AsItStates,
    );
    gemv.check([Product::of(F16).m(1).n(128).k(128)]);
}
