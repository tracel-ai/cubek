//! Correctness over the conv2d benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_convolution::Strategy;
use cubek_convolution::eval::benchmarks::{Conv2dCorrectness, Conv2dProblem};
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

/// Loose epsilon for f16 conv reductions; tighten if needed.
const F16_EPS: f32 = 0.5;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    use cubek_convolution::eval::benchmarks::{problems, strategies};

    let strategy: Strategy = lookup(strategies(), strategy_id);
    let problem: Conv2dProblem = lookup(problems(), problem_id);

    let actual = match Conv2dCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = Conv2dCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
#[ignore = "slow CPU reference (large kernel)"]
fn alexnet_like() {
    // Pick the first strategy in the catalogue at runtime so this stays
    // robust to renames; if a strategy id changes we'll see a panic listing
    // the available options.
    use cubek_convolution::eval::benchmarks::strategies;
    let id = strategies()
        .first()
        .map(|s| s.id.clone())
        .expect("conv2d strategies catalogue is empty");
    run(&id, "alexnet_like");
}

#[test]
#[ignore = "very slow CPU reference (large_kernel)"]
fn large_kernel() {
    use cubek_convolution::eval::benchmarks::strategies;
    let id = strategies()
        .first()
        .map(|s| s.id.clone())
        .expect("conv2d strategies catalogue is empty");
    run(&id, "large_kernel");
}
