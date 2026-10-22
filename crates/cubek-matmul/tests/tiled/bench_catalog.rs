//! Correctness over the tiled entries of the gemm catalogue.

#![cfg(feature = "benchmarks")]

use cubek_matmul::{
    eval::benchmarks::gemm::{GemmCorrectness, GemmProblem},
    strategy::Strategy,
};
use cubek_test_utils::{
    CatalogEntry, Correctness, TestOutcome, assert_equals_approx, skip_unless_cpu,
};

const SEEDS: [u64; 2] = [12, 34];

/// Loose epsilon. f16 matmul reductions accumulate noise quickly; tighten if
/// you ever need this catalogue to gate on precision regressions.
const F16_EPS: f32 = 0.5;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run_gemm(strategy_id: &str, problem_id: &str) {
    use cubek_matmul::eval::benchmarks::gemm::{problems, strategies};

    let strategy: Strategy = lookup(strategies(), strategy_id);
    let problem: GemmProblem = lookup(problems(), problem_id);

    let actual = match GemmCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = GemmCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

/// CpuGemm through the benchmark-catalog path (vs the extended tests' `test_matmul_strategy`
/// path). `vecmat` keeps the CPU reference cheap (`m = 1`).
#[test]
fn gemm_cpu_gemm_vecmat_2x1x4096x4096_rr_f32() {
    let client = cubecl::test_device().client();
    if skip_unless_cpu(&client) {
        return;
    }
    run_gemm("cpu_gemm", "vecmat_2x1x4096x4096_rr_f32");
}
