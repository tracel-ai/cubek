//! Correctness over the interpolate benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_interpolate::definition::InterpolateProblem;
use cubek_interpolate::eval::benchmarks::{InterpolateCorrectness, InterpolateStrategy};
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

const INTERP_EPS: f32 = 1e-3;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    use cubek_interpolate::eval::benchmarks::{problems, strategies};

    let strategy: InterpolateStrategy = lookup(strategies(), strategy_id);
    let problem: InterpolateProblem = lookup(problems(), problem_id);

    let actual = match InterpolateCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = InterpolateCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, INTERP_EPS)
        .as_test_outcome()
        .enforce();
}

const STRATEGY: &str = "default";

#[test]
fn nearest_downsample_default() {
    run(
        STRATEGY,
        "NEAREST_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
    );
}

#[test]
fn bilinear_downsample_default() {
    run(
        STRATEGY,
        "BILINEAR_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
    );
}

#[test]
fn bicubic_downsample_default() {
    run(
        STRATEGY,
        "BICUBIC_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
    );
}

#[test]
fn lanczos3_downsample_default() {
    run(
        STRATEGY,
        "LANCZOS3_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
    );
}

#[test]
fn nearest_backward_downsample_default() {
    run(
        STRATEGY,
        "NEAREST_BACKWARD_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
    );
}
