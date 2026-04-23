//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, square};
use crate::suite::test_matmul_strategy;

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleCyclicCmma(Default::default()),
    );
}

#[test]
fn simple_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleCyclicMma(Default::default()),
    );
}

#[test]
fn simple_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleStridedCmma(Default::default()),
    );
}

#[test]
fn simple_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleStridedMma(Default::default()),
    );
}

#[test]
fn simple_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTilewiseCmma(Default::default()),
    );
}

#[test]
fn simple_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTilewiseMma(Default::default()),
    );
}

#[test]
fn simple_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncStridedCmma(Default::default()),
    );
}

#[test]
fn simple_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncStridedMma(Default::default()),
    );
}

#[test]
fn simple_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncCyclicCmma(Default::default()),
    );
}

#[test]
fn simple_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncCyclicMma(Default::default()),
    );
}

#[test]
fn double_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleCyclicCmma(Default::default()),
    );
}

#[test]
fn double_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleCyclicMma(Default::default()),
    );
}

#[test]
fn double_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTilewiseCmma(Default::default()),
    );
}

#[test]
fn double_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTilewiseMma(Default::default()),
    );
}

#[test]
fn double_hybrid_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleHybridCmma(Default::default()),
    );
}

#[test]
fn double_hybrid_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleHybridMma(Default::default()),
    );
}

#[test]
fn double_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncCyclicCmma(Default::default()),
    );
}

#[test]
fn double_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncCyclicMma(Default::default()),
    );
}

#[test]
fn double_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncStridedCmma(Default::default()),
    );
}

#[test]
fn double_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncStridedMma(Default::default()),
    );
}

#[test]
fn specialized_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedCyclicCmma(Default::default()),
    );
}

#[test]
fn specialized_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedCyclicMma(Default::default()),
    );
}

#[test]
fn specialized_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedStridedCmma(Default::default()),
    );
}

#[test]
fn specialized_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedStridedMma(Default::default()),
    );
}

#[test]
fn ordered_double_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::OrderedDoubleCmma(Default::default()),
    );
}

#[test]
fn ordered_double_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::OrderedDoubleMma(Default::default()),
    );
}
