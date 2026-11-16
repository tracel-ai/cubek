//! Inferred-blueprint smoke tests for unit-based routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, Product};

#[cfg(feature = "heavy")]
#[test]
fn simple_unit() {
    let unit = ByStrategy::new(
        MultiLevel::SimpleUnit(Default::default()),
        Declines::AsItStates,
    );
    unit.check([Product::of(F16).m(64).n(64).k(64)]);
}

#[cfg(feature = "heavy")]
#[test]
fn double_unit() {
    let unit = ByStrategy::new(
        MultiLevel::DoubleUnit(Default::default()),
        Declines::AsItStates,
    );
    unit.check([Product::of(F16).m(64).n(64).k(64)]);
}
