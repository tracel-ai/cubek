//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use crate::harness::{client, f16_elems, square, test_matmul_strategy};

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn simple_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridCmma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridMma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicMma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedMma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleCmma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleMma(Default::default()).into(),
    );
}
