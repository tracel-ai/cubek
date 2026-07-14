//! Out-of-bounds tail tier: non-causal, unmasked problems whose dimensions do
//! not divide the tiling spans, so correctness rests entirely on the OOB
//! predicate — the tail tiles of each axis must contribute exactly nothing.
//!
//! Inputs are drawn at `uniform(-8, 8)` as well as the default unit range: a
//! leak of a real (unmasked) score or value grows with input magnitude, while
//! a zero/garbage leak does not, so the two ranges also fingerprint the
//! failure mode. Aligned control cases pin the same strategies on shapes with
//! no tail at all.

use crate::attention::forward::launcher::{test_launch, test_launch_permuted, test_launch_scaled};
use cubecl::{
    Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive, ir::AddressType,
};
use cubek_attention::{
    forward::definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions,
        AttentionProblem,
    },
    forward::launch::{BlueprintStrategy, Strategy},
    forward::routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

const RANGE: f32 = 8.0;
const RANGE_EPSILON: f32 = 1e-2;

fn f32_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_float_dtype(
        f32::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(client),
    )
}

fn unit_inferred() -> Strategy {
    Strategy::Unit(BlueprintStrategy::Inferred(()))
}

fn blackbox_accelerated_inferred() -> Strategy {
    Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
        num_planes: 1,
        seq_q: 1,
        seq_kv: 1,
    }))
}

fn problem(
    global_dtypes: AttentionGlobalTypes,
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
) -> AttentionProblem {
    problem_causal(global_dtypes, (seq_q, seq_kv, head_dim, val_dim), false)
}

fn problem_heads(
    global_dtypes: AttentionGlobalTypes,
    num_heads: usize,
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
) -> AttentionProblem {
    problem_heads_causal(
        global_dtypes,
        num_heads,
        (seq_q, seq_kv, head_dim, val_dim),
        false,
    )
}

fn problem_heads_causal(
    global_dtypes: AttentionGlobalTypes,
    num_heads: usize,
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
    causal: bool,
) -> AttentionProblem {
    let mut problem = problem_causal(global_dtypes, (seq_q, seq_kv, head_dim, val_dim), causal);
    problem.dims.num_heads = num_heads;
    problem
}

fn problem_causal(
    global_dtypes: AttentionGlobalTypes,
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
    causal: bool,
) -> AttentionProblem {
    AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

// Aligned controls: no tail on any axis, so a failure here means the strategy
// is wrong independently of the OOB predicate.

#[test]
fn aligned_512_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f32_dtypes(&client), (512, 512, 64, 64)),
        unit_inferred(),
    )
}

#[test]
fn aligned_512_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (512, 512, 64, 64)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

#[test]
#[ignore = "BlackboxAcceleratedRoutine hardcodes f16 tiles, so f32 problems lose precision with \
            input magnitude; run once blackbox tile dtypes follow the global dtype"]
fn aligned_512_blackbox_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (512, 512, 64, 64)),
        blackbox_accelerated_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

// The literal Whisper-encoder shape: 1500 is unaligned on both sequence axes.

#[test]
fn square_1500_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f32_dtypes(&client), (1500, 1500, 64, 64)),
        unit_inferred(),
    )
}

#[test]
fn square_1500_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (1500, 1500, 64, 64)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

// Axis splits: only one of the two sequence axes carries a tail.

#[test]
fn kv_tail_only_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (1024, 1500, 64, 64)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

#[test]
fn q_tail_only_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (1500, 1024, 64, 64)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

#[test]
#[ignore = "BlackboxAcceleratedRoutine hardcodes f16 tiles, so f32 problems lose precision with \
            input magnitude; run once blackbox tile dtypes follow the global dtype"]
fn kv_tail_blackbox_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (32, 1500, 64, 64)),
        blackbox_accelerated_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

// Encoder-decoder shapes (Whisper's decoder): the query sequence is far
// smaller than one q-stage span, so every unit's rows beyond seq_q are
// entirely out of bounds — cross-attention prefill (few queries against a
// long encoder), single-query decode, and a short causal prefill.

#[test]
fn cross_attention_prefill_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f32_dtypes(&client), (4, 1500, 64, 64)),
        unit_inferred(),
    )
}

#[test]
fn single_query_decode_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f32_dtypes(&client), (1, 1500, 64, 64)),
        unit_inferred(),
    )
}

#[test]
fn causal_short_prefill_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem_causal(f32_dtypes(&client), (4, 4, 64, 64), true),
        unit_inferred(),
    )
}

// Permuted inputs: q/k/v as `(b, heads, seq, hd)` views of `(b, seq, heads,
// hd)` buffers — the layout every attention module produces from its fused
// projection. Needs `heads >= 2` (with one head the permutation degenerates
// to the contiguous layout).

#[test]
fn permuted_inputs_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_permuted(
        client.clone(),
        problem_heads(f32_dtypes(&client), 2, (64, 64, 32, 32)),
        unit_inferred(),
    )
}

#[test]
fn permuted_inputs_encoder_shape_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_permuted(
        client.clone(),
        problem_heads(f32_dtypes(&client), 6, (1500, 1500, 64, 64)),
        unit_inferred(),
    )
}

#[test]
fn permuted_inputs_causal_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_permuted(
        client.clone(),
        problem_heads_causal(f32_dtypes(&client), 2, (64, 64, 32, 32), true),
        unit_inferred(),
    )
}

#[test]
fn permuted_inputs_causal_encoder_shape_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_permuted(
        client.clone(),
        problem_heads_causal(f32_dtypes(&client), 6, (1500, 1500, 64, 64), true),
        unit_inferred(),
    )
}

#[test]
fn permuted_inputs_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_permuted(
        client.clone(),
        problem_heads(f32_dtypes(&client), 2, (64, 64, 32, 32)),
        blackbox_accelerated_inferred(),
    )
}

// Minimal tails: the smallest shapes that still exercise a partial tile on
// exactly one axis, cheap enough to bisect against forced blueprints.

#[test]
fn kv_tail_minimal_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (8, 43, 32, 32)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}

#[test]
fn q_tail_minimal_unit_large_magnitude() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_scaled(
        client.clone(),
        problem(f32_dtypes(&client), (300, 64, 32, 32)),
        unit_inferred(),
        RANGE,
        Some(RANGE_EPSILON),
    )
}
