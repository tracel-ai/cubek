//! Correctness over the reduce benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_reduce::ReduceStrategy;
use cubek_reduce::eval::benchmarks::{ReduceBenchPrecision, ReduceCorrectness, ReduceProblem};
use cubek_reduce::eval::cpu_reference::comparison_epsilon;
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

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

    assert_equals_approx(&actual, &expected, comparison_epsilon(problem.config))
        .as_test_outcome()
        .enforce();
}

/// The catalogue only offers f16 rows where the device can fold them, so a
/// runtime without f16 has no `_f16` id to look up.
fn run_f16(strategy_id: &str, problem_id: &str) {
    use cubek_reduce::eval::benchmarks::precisions;

    if !precisions().contains(&ReduceBenchPrecision::F16) {
        return;
    }
    run(strategy_id, problem_id);
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

// The fused entries get timed against the two-launch ones, so they have to be
// correct at the benchmark's own shape and strategy, not only at the small
// shapes the integration tests cover.
#[test]
fn topk2_fused_axis2_32x512x4095_cube_serial() {
    run("cube_serial", "topk2_fused_axis2_32x512x4095");
}

#[test]
fn topk3_fused_axis2_32x512x4095_cube_serial() {
    run("cube_serial", "topk3_fused_axis2_32x512x4095");
}

#[test]
fn topk3_fused_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "topk3_fused_axis2_32x512x4095");
}

#[test]
fn topk3_two_launch_axis2_32x512x4095_cube_serial() {
    run("cube_serial", "topk3_two_launch_axis2_32x512x4095");
}

#[test]
fn max_fused_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "max_fused_axis2_32x512x4095");
}

#[test]
fn max_fused_axis2_32x512x4095_cube_serial() {
    run("cube_serial", "max_fused_axis2_32x512x4095");
}

#[test]
fn min_fused_axis2_32x512x4095_unit_parallel() {
    run("unit_parallel", "min_fused_axis2_32x512x4095");
}

#[test]
fn min_fused_axis2_32x512x4095_cube_serial() {
    run("cube_serial", "min_fused_axis2_32x512x4095");
}

#[test]
fn sum_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "sum_axis2_32x512x4095_f16");
}

#[test]
fn sum_axis2_32x512x4095_f16_plane_parallel() {
    run_f16("plane_parallel", "sum_axis2_32x512x4095_f16");
}

#[test]
fn arg_topk1_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "arg_topk1_axis2_32x512x4095_f16");
}

#[test]
fn arg_topk2_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "arg_topk2_axis2_32x512x4095_f16");
}

#[test]
fn arg_topk3_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "arg_topk3_axis2_32x512x4095_f16");
}

#[test]
fn topk2_fused_axis2_32x512x4095_f16_cube_serial() {
    run_f16("cube_serial", "topk2_fused_axis2_32x512x4095_f16");
}

#[test]
fn topk3_fused_axis2_32x512x4095_f16_cube_serial() {
    run_f16("cube_serial", "topk3_fused_axis2_32x512x4095_f16");
}

#[test]
fn topk3_fused_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "topk3_fused_axis2_32x512x4095_f16");
}

#[test]
fn topk3_two_launch_axis2_32x512x4095_f16_cube_serial() {
    run_f16("cube_serial", "topk3_two_launch_axis2_32x512x4095_f16");
}

#[test]
fn max_fused_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "max_fused_axis2_32x512x4095_f16");
}

#[test]
fn max_fused_axis2_32x512x4095_f16_cube_serial() {
    run_f16("cube_serial", "max_fused_axis2_32x512x4095_f16");
}

#[test]
fn min_fused_axis2_32x512x4095_f16_unit_parallel() {
    run_f16("unit_parallel", "min_fused_axis2_32x512x4095_f16");
}

#[test]
fn min_fused_axis2_32x512x4095_f16_cube_serial() {
    run_f16("cube_serial", "min_fused_axis2_32x512x4095_f16");
}
