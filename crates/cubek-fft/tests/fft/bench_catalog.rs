//! Correctness over the FFT benchmark catalogue.
#![cfg(feature = "benchmarks")]

use cubek_fft::eval::benchmarks::{FftCorrectness, FftProblem, FftStrategy};
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

/// FFT is f32 round-trip, so we can be tighter than the matmul side.
const FFT_EPS: f32 = 1e-3;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    use cubek_fft::eval::benchmarks::{problems, strategies};

    let strategy: FftStrategy = lookup(strategies(), strategy_id);
    let problem: FftProblem = lookup(problems(), problem_id);

    let actual = match FftCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = FftCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, FFT_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
fn forward_5x2x2048_default() {
    run("default", "forward_5x2x2048");
}

#[test]
fn inverse_5x2x2048_default() {
    run("default", "inverse_5x2x2048");
}

#[test]
fn forward_1x4096_default() {
    run("default", "forward_1x4096");
}

#[test]
fn forward_1x16384_default() {
    run("default", "forward_1x16384");
}
