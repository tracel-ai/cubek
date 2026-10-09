//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_test_utils::{TestOutcome, ValidationResult};

use crate::harness::{
    client, f16_elems, f32_elems, rect, run_with_strides, square, test_matmul_strategy,
};

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicCmma(Default::default()).into(),
    );
}

/// A small f32 problem runs rather than declining: a routine that runs at 37 rows also runs at 32,
/// since autotune buckets the two together. Asserted as executed on a device with cmma, where the
/// default test policy would accept the decline; reported as skipped on one without.
#[test]
fn simple_cyclic_cmma_runs_a_small_f32_problem() {
    let client = client();
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device offers no cmma instruction".to_string(),
        ))
        .enforce();
        return;
    }
    let outcome = run_with_strides(
        client,
        rect(32, 32, 64, f32_elems()),
        MultiLevel::SimpleCyclicCmma(Default::default()).into(),
    );
    assert!(
        matches!(outcome, TestOutcome::Validated(ValidationResult::Pass)),
        "expected the matmul to execute and validate, got {outcome:?}"
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

#[test]
fn simple_cyclic_cmma_small_k_and_partial_line_width_rows_stay_correct() {
    use crate::harness::{f32_elems, rect};
    for _ in 0..20 {
        for (m, n, k) in [(382, 10, 1), (382, 10, 3), (4096, 14, 5)] {
            test_matmul_strategy(
                client(),
                rect(m, n, k, f32_elems()),
                MultiLevel::SimpleCyclicCmma(Default::default()).into(),
            );
        }
    }
}
