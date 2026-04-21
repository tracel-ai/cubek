//! Inferred-blueprint smoke tests for the plane vec-mat inner-product routines.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, rect};
use crate::suite::test_matmul_strategy;

#[test]
fn simple_vecmat() {
    // 1xNxK shape — the routine is designed for vec-mat.
    test_matmul_strategy(
        client(),
        rect(1, 256, 256, f16_elems()),
        Strategy::SimpleVecMat(Default::default()),
    );
}

#[test]
fn double_vecmat() {
    test_matmul_strategy(
        client(),
        rect(1, 256, 256, f16_elems()),
        Strategy::DoubleVecMat(Default::default()),
    );
}
