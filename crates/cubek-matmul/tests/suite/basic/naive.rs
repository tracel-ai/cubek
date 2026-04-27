//! Smoke tests for the naive routine.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, f32_elems, rect, square};
use crate::suite::test_matmul_strategy;

#[test]
fn naive_small_f16() {
    test_matmul_strategy(client(), square(16, f16_elems()), Strategy::Naive);
}

#[cfg(feature = "basic")]
#[test]
fn naive_medium_f32() {
    test_matmul_strategy(client(), square(128, f32_elems()), Strategy::Naive);
}

#[test]
fn naive_odd_shape() {
    test_matmul_strategy(client(), rect(1, 255, 101, f16_elems()), Strategy::Naive);
}
