//! Inferred-blueprint smoke tests for unit-based routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use crate::harness::{client, f16_elems, square, test_matmul_strategy};

#[cfg(feature = "heavy")]
#[test]
fn simple_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        MultiLevel::SimpleUnit(Default::default()).into(),
    );
}

#[cfg(feature = "heavy")]
#[test]
fn double_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        MultiLevel::DoubleUnit(Default::default()).into(),
    );
}
