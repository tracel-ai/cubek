//! Correctness over the reduce benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_reduce::ReduceStrategy;
use cubek_reduce::eval::benchmarks::{ReduceCorrectness, ReduceProblem};
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

/// f32 reductions over ~tens of millions of elements; some kernels accumulate
/// noticeable noise. Tightened tolerances belong in the existing per-routine
/// integration tests.
const REDUCE_EPS: f32 = 1.0;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    use cubek_reduce::eval::benchmarks::{problems, strategies};

    let strategy: ReduceStrategy = lookup(strategies(), strategy_id);
    let problem: ReduceProblem = lookup(problems(), problem_id);

    let actual = match ReduceCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = ReduceCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, REDUCE_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
fn sum_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "sum_axis2_32x512x4095");
}

#[test]
fn sum_axis2_32x512x4095_plane_parallel() {
    run("plane_parallel", "sum_axis2_32x512x4095");
}

#[test]
fn arg_topk1_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "arg_topk1_axis2_32x512x4095");
}

#[test]
fn arg_topk2_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "arg_topk2_axis2_32x512x4095");
}

#[test]
fn arg_topk3_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "arg_topk3_axis2_32x512x4095");
}
