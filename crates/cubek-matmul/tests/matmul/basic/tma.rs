//! Inferred-blueprint smoke tests for TMA routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use super::common::{client, f16_elems, square};
use crate::matmul::test_matmul_strategy;

#[test]
fn simple_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTmaCmma(Default::default()).into(),
    );
}

#[test]
fn simple_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTmaMma(Default::default()).into(),
    );
}

#[test]
fn double_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTmaCmma(Default::default()).into(),
    );
}

#[test]
fn double_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTmaMma(Default::default()).into(),
    );
}

#[test]
fn specialized_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedTmaCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedTmaMma(Default::default()).into(),
    );
}
