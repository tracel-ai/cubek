//! Inferred-blueprint smoke tests for TMA routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, Product};

#[test]
fn simple_tma_cmma() {
    let tma = ByStrategy::new(
        MultiLevel::SimpleTmaCmma(Default::default()),
        Declines::AsItStates,
    );
    tma.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn simple_tma_mma() {
    let tma = ByStrategy::new(
        MultiLevel::SimpleTmaMma(Default::default()),
        Declines::AsItStates,
    );
    tma.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn specialized_tma_cmma() {
    let tma = ByStrategy::new(
        MultiLevel::SpecializedTmaCmma(Default::default()),
        Declines::AsItStates,
    );
    tma.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn specialized_tma_mma() {
    let tma = ByStrategy::new(
        MultiLevel::SpecializedTmaMma(Default::default()),
        Declines::AsItStates,
    );
    tma.check([Product::of(F16).m(256).n(256).k(256)]);
}
