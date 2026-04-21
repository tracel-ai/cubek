//! Inferred-blueprint smoke tests for unit-based routines, plus the interleaved
//! routine (exposed via [`TestStrategy`] because it is not part of the public
//! `Strategy` enum).

use cubek_matmul::launch::{Strategy, test_only::TestStrategy};

use super::common::{client, f16_elems, square};
use crate::suite::{test_matmul_strategy, test_matmul_test_strategy};

#[test]
fn simple_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        Strategy::SimpleUnit(Default::default()),
    );
}

#[test]
fn double_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        Strategy::DoubleUnit(Default::default()),
    );
}

#[test]
fn interleaved() {
    test_matmul_test_strategy(
        client(),
        square(64, f16_elems()),
        TestStrategy::Interleaved(Default::default()),
    );
}
