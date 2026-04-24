//! Inferred-blueprint smoke tests for TMA routines.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, square};
use crate::suite::test_matmul_strategy;

#[test]
fn simple_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTmaCmma(Default::default()),
    );
}

#[test]
fn simple_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTmaMma(Default::default()),
    );
}

#[test]
fn double_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTmaCmma(Default::default()),
    );
}

#[test]
fn double_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTmaMma(Default::default()),
    );
}

#[test]
fn specialized_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedTmaCmma(Default::default()),
    );
}

#[test]
fn specialized_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedTmaMma(Default::default()),
    );
}
