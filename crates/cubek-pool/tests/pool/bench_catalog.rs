//! Correctness over the pool benchmark catalogue.
//!
//! Runs each `(strategy, problem)` pair from
//! [`cubek_pool::eval::benchmarks::strategies`] × [`problems`] through the kernel
//! and compares against the CPU reference.

#![cfg(feature = "benchmarks")]

use cubek_pool::definition::{PoolMode, PoolProblem};
use cubek_pool::eval::benchmarks::{PoolCorrectness, problems, strategies};
use cubek_test_utils::{Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

fn run_pair(strategy_id: &str, problem_id: &str) {
    let strategy = strategies()
        .into_iter()
        .find(|s| s.id == strategy_id)
        .unwrap_or_else(|| panic!("unknown strategy: {strategy_id}"))
        .value;
    let spec = problems()
        .into_iter()
        .find(|p| p.id == problem_id)
        .unwrap_or_else(|| panic!("unknown problem: {problem_id}"))
        .value;

    let actual = match PoolCorrectness.kernel_result(&strategy, &spec, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = PoolCorrectness
        .reference_result(&spec, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}/{strategy_id}: {e}"));

    let eps = tolerance_for(&spec);
    assert_equals_approx(&actual, &expected, eps)
        .as_test_outcome()
        .enforce();
}

fn tolerance_for(problem: &PoolProblem) -> f32 {
    match problem {
        PoolProblem::Forward(p) => match p {
            cubek_pool::definition::PoolForward::D2(spec) => match &spec.mode {
                PoolMode::Max(_) => 0.0,
                _ => 1e-5,
            },
            _ => 1e-5,
        },
        PoolProblem::Backward(p) => match p {
            cubek_pool::definition::PoolBackward::D2(spec) => match &spec.mode {
                PoolMode::Max(_) => 0.0,
                _ => 1e-5,
            },
            _ => 1e-5,
        },
    }
}

const STRATEGY: &str = "default";

#[test]
#[ignore = "slow CPU reference"]
fn max_pool2d_forward_resnet_init() {
    run_pair(STRATEGY, "MAX_POOL2D_FWD_RESNET_INIT");
}

#[test]
#[ignore = "slow CPU reference"]
fn max_pool2d_backward_resnet_init() {
    run_pair(STRATEGY, "MAX_POOL2D_BWD_RESNET_INIT");
}

#[test]
#[ignore = "very slow CPU reference (deep)"]
fn max_pool2d_forward_deep() {
    run_pair(STRATEGY, "MAX_POOL2D_FWD_DEEP");
}

#[test]
#[ignore = "very slow CPU reference (deep)"]
fn max_pool2d_backward_deep() {
    run_pair(STRATEGY, "MAX_POOL2D_BWD_DEEP");
}

#[test]
#[ignore = "slow CPU reference (throughput)"]
fn avg_pool2d_forward_throughput() {
    run_pair(STRATEGY, "AVG_POOL2D_FWD_THROUGHPUT");
}

#[test]
#[ignore = "slow CPU reference (throughput)"]
fn avg_pool2d_backward_throughput() {
    run_pair(STRATEGY, "AVG_POOL2D_BWD_THROUGHPUT");
}

#[test]
#[ignore = "slow CPU reference (global)"]
fn adaptive_avg_pool2d_forward_global() {
    run_pair(STRATEGY, "ADAPTIVE_AVG_POOL2D_FWD_GLOBAL");
}

#[test]
#[ignore = "slow CPU reference (global)"]
fn adaptive_avg_pool2d_backward_global() {
    run_pair(STRATEGY, "ADAPTIVE_AVG_POOL2D_BWD_GLOBAL");
}

#[test]
#[ignore = "slow CPU reference (reduce)"]
fn adaptive_avg_pool2d_forward_reduce() {
    run_pair(STRATEGY, "ADAPTIVE_AVG_POOL2D_FWD_REDUCE");
}

#[test]
#[ignore = "slow CPU reference (reduce)"]
fn adaptive_avg_pool2d_backward_reduce() {
    run_pair(STRATEGY, "ADAPTIVE_AVG_POOL2D_BWD_REDUCE");
}
