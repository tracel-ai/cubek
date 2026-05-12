//! Correctness over the matmul benchmark catalogues (gemm + gemv).

#![cfg(feature = "benchmarks")]

use cubek_matmul::eval::benchmarks::gemm::{GemmCorrectness, GemmProblem};
use cubek_matmul::eval::benchmarks::gemv::{GemvCorrectness, GemvProblem};
use cubek_matmul::launch::Strategy;
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

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

fn run_gemv(strategy_id: &str, problem_id: &str) {
    use cubek_matmul::eval::benchmarks::gemv::{problems, strategies};

    let strategy: Strategy = lookup(strategies(), strategy_id);
    let problem: GemvProblem = lookup(problems(), problem_id);

    let actual = match GemvCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = GemvCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
fn gemm_rect_1x512x512x512_rr_f16() {
    run_gemm("simple_cyclic_cmma", "rect_1x512x512x512_rr_f16");
}

#[test]
#[ignore = "slow CPU reference + CMMA fallbacks"]
fn gemm_square_2x1024_rr_f16() {
    run_gemm("simple_cyclic_cmma", "square_2x1024_rr_f16");
}

#[test]
#[ignore = "very slow CPU reference"]
fn gemm_square_1x6144_rr_f16() {
    run_gemm("simple_cyclic_cmma", "square_1x6144_rr_f16");
}

#[test]
fn gemv_vecmat_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "vecmat_b2_out4096_k8192_rr");
}

#[test]
fn gemv_matvec_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "matvec_b2_out4096_k8192_rr");
}
