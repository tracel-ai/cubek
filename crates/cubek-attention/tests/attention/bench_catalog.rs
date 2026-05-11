//! Correctness over the attention benchmark catalogue.
//!
//! Runs each `(strategy, problem)` pair from
//! [`cubek_attention::eval::benchmarks::strategies`] × [`problems`] through the kernel
//! and compares against the CPU flash-attention v2 reference.

#![cfg(feature = "benchmarks")]

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek_attention::definition::{AttentionElems, AttentionGlobalTypes, AttentionProblem};
use cubek_attention::eval::benchmarks::{
    AttentionCorrectness, AttentionSpec, problems, strategies,
};
use cubek_attention::eval::cpu_reference::attention_epsilon;
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

    // Route kernel failures (e.g. "no mma instruction available on this
    // backend") through `TestOutcome::CompileError` so the active
    // `cubek.toml` test policy decides whether to skip or fail — same
    // contract as the existing launcher tests.
    let actual = match AttentionCorrectness.kernel_result(&strategy, &spec, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = AttentionCorrectness
        .reference_result(&spec, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}/{strategy_id}: {e}"));

    let elems = elems_for(&spec);
    let eps = attention_epsilon(&elems, 0.01);
    assert_equals_approx(&actual, &expected, eps)
        .as_test_outcome()
        .enforce();
}

/// Mirror `AttentionCorrectness::kernel_result`'s dtype setup so the epsilon
/// matches the precision the kernel actually ran with.
fn elems_for(spec: &AttentionSpec) -> AttentionElems {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(&client),
    );
    let problem = AttentionProblem {
        dims: spec.dims.clone(),
        global_dtypes,
        masked: spec.masked,
        options: spec.options.clone(),
        address_type: Default::default(),
    };
    AttentionElems::from_global_types(
        &problem.global_dtypes,
        half::f16::as_type_native_unchecked().storage_type(),
        &problem.options.accumulator_precision,
    )
}

const STRATEGY: &str = "blackbox_accelerated_inferred";

#[test]
fn bert_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "bert");
}

#[test]
fn gpt2_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "gpt2");
}

/// Fast regression for the causal+materialized-mask absolute-row bug. Real
/// GPT-2 takes ~50s; this 64×64 variant catches the same kernel path in
/// under a second.
#[test]
fn gpt2_tiny_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "gpt2_tiny");
}

#[test]
#[ignore = "slow CPU reference (~minutes)"]
fn llama_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "llama");
}

#[test]
#[ignore = "slow CPU reference (~tens of minutes)"]
fn long_context_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "long_context");
}

#[test]
fn encoder_decoder_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "encoder_decoder");
}

#[test]
#[ignore = "slow CPU reference (~minutes)"]
fn mask_causal_4096_blackbox_accelerated_inferred() {
    run_pair(STRATEGY, "mask_causal_4096");
}
