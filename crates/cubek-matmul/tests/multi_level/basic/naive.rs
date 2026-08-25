//! Smoke tests for the naive routine.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use crate::harness::{client, f16_elems, f32_elems, rect, square, test_matmul_strategy};

#[test]
fn naive_small_f16() {
    test_matmul_strategy(client(), square(16, f16_elems()), MultiLevel::Naive.into());
}

#[cfg(feature = "heavy")]
#[test]
fn naive_medium_f32() {
    test_matmul_strategy(client(), square(128, f32_elems()), MultiLevel::Naive.into());
}

#[test]
fn naive_odd_shape() {
    test_matmul_strategy(
        client(),
        rect(1, 255, 101, f16_elems()),
        MultiLevel::Naive.into(),
    );
}
