//! Correctness over the matmul benchmark catalogues (gemm + gemv).

#![cfg(feature = "benchmarks")]

use cubecl::Runtime;
use cubek_matmul::eval::benchmarks::gemm::{GemmCorrectness, GemmProblem};
use cubek_matmul::eval::benchmarks::gemv::{GemvCorrectness, GemvProblem};
use cubek_matmul::strategy::Strategy;
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

/// CpuGemm through the benchmark-catalog path (vs the extended tests' `test_matmul_strategy`
/// path). `vecmat` keeps the CPU reference cheap (`m = 1`).
#[test]
fn gemm_cpu_gemm_vecmat_2x1x4096x4096_rr_f32() {
    let client = cubecl::TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    run_gemm("cpu_gemm", "vecmat_2x1x4096x4096_rr_f32");
}

#[test]
fn gemv_vecmat_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "vecmat_b2_out4096_k8192_rr");
}

#[test]
fn gemv_matvec_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "matvec_b2_out4096_k8192_rr");
}

/// Timing probe: the tile-DSL cyclic cmma vs the legacy SimpleAlgorithm it ports.
/// Run manually: `cargo test-metal-benchmark gemm_cmma_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing probe, run manually"]
fn gemm_cmma_timing_vs_legacy() {
    use cubek_matmul::eval::benchmarks::gemm::{bench, problems, strategies};

    let problem: GemmProblem = lookup(problems(), "square_2x4096_rr_f16");
    for id in ["cmma", "simple_cyclic_cmma"] {
        let strategy: Strategy = lookup(strategies(), id);
        let samples = bench(&strategy, &problem, 10).unwrap();
        let mut ds = samples.durations.clone();
        ds.sort();
        println!(
            "{id}: median {:?} over {} samples, {:.2} TFLOPS",
            ds[ds.len() / 2],
            ds.len(),
            samples.tflops.unwrap_or(0.0)
        );
    }
}

/// Print the backend's cmma configs (debugging aid).
#[test]
#[ignore = "debug probe"]
fn print_cmma_configs() {
    let client = cubecl::TestRuntime::client(&Default::default());
    for c in client.properties().features.matmul.cmma.iter() {
        println!("{:?}", c);
    }
}
