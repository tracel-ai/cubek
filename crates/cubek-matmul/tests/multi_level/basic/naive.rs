//! Smoke tests for the naive routine.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, F32, Product};

#[test]
fn naive_small_f16() {
    let naive = ByStrategy::new(MultiLevel::Naive, Declines::AsItStates);
    naive.check([Product::of(F16).m(16).n(16).k(16)]);
}

#[cfg(feature = "heavy")]
#[test]
fn naive_medium_f32() {
    let naive = ByStrategy::new(MultiLevel::Naive, Declines::AsItStates);
    naive.check([Product::of(F32).m(128).n(128).k(128)]);
}

#[test]
fn naive_odd_shape() {
    let naive = ByStrategy::new(MultiLevel::Naive, Declines::AsItStates);
    naive.check([Product::of(F16).m(1).n(255).k(101)]);
}
