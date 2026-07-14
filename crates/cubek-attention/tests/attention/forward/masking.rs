//! Masking acceptance tier: causal and materialized-mask semantics, pinned at
//! the problem level (tensors in, tensors out, checked against the CPU
//! reference) so every case survives a routine/architecture swap unchanged.
//!
//! Contract: the materialized mask is a boolean predicate tensor (nonzero =
//! masked), not an additive bias. Fully-masked rows emit exactly zero.

use crate::attention::forward::assert_result;
use crate::attention::forward::launcher::test_launch;
use cubecl::{
    Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive, ir::AddressType,
    zspace::Shape,
};
use cubek_attention::forward::definition::{
    AccumulatorPrecision, AttentionDims, AttentionElems, AttentionGlobalTypes, AttentionIdent,
    AttentionOptions, AttentionProblem,
};
use cubek_attention::forward::launch::{BlueprintStrategy, Strategy, launch_ref};
use cubek_attention::forward::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, launch_and_capture_outcome,
};

fn f16_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
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
    causal: bool,
    masked: bool,
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
        masked,
        global_dtypes,
        options: AttentionOptions {
            causal,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

// Causal: square and both rectangular orientations. Rectangular causal is
// bottom-right aligned (`j > i + seq_kv - seq_q`), matching burn's fallback
// and the KV-cache decode contract: the last query row always attends the
// whole key sequence.

#[test]
fn causal_square_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_square_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), true, false),
        blackbox_accelerated_inferred(),
    )
}

#[test]
fn causal_rect_q_lt_kv_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (32, 128, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_rect_q_lt_kv_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (32, 128, 32, 32), true, false),
        blackbox_accelerated_inferred(),
    )
}

#[test]
fn causal_rect_q_gt_kv_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (128, 32, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_rect_q_gt_kv_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (128, 32, 32, 32), true, false),
        blackbox_accelerated_inferred(),
    )
}

// Cached-decode shapes: with bottom-right alignment the single query row of a
// decode step attends every cached key (top-left alignment would wrongly
// restrict it to the first key), and a short continuation block attends the
// whole prefix plus its own causal triangle.

#[test]
fn causal_decode_single_query_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (1, 128, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_cached_prefill_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (4, 128, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_cached_prefill_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (16, 128, 32, 32), true, false),
        blackbox_accelerated_inferred(),
    )
}

// Causal on shapes that are not multiples of the tile sizes: exercises the
// out-of-bounds predicate together with the causal one (padding must be -inf
// at the softmax, zero at the matmuls).

#[test]
fn causal_odd_shapes_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (20, 44, 32, 32), true, false),
        unit_inferred(),
    )
}

#[test]
fn causal_odd_shapes_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (20, 44, 32, 32), true, false),
        blackbox_accelerated_inferred(),
    )
}

// Materialized boolean mask (bernoulli), alone and combined with causal.

#[test]
fn masked_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), false, true),
        unit_inferred(),
    )
}

#[test]
fn masked_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), false, true),
        blackbox_accelerated_inferred(),
    )
}

#[test]
fn masked_causal_combined_unit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), true, true),
        unit_inferred(),
    )
}

#[test]
fn masked_causal_combined_blackbox() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), (64, 64, 32, 32), true, true),
        blackbox_accelerated_inferred(),
    )
}

/// Crafted mask hitting the numeric edge cases bernoulli can't reach:
/// - rows 0..4 fully masked: the output row must be exactly 0.0 (the
///   `l = 0` guard, not just approximately small);
/// - row 4 masked on the first kv half: the running max starts at -inf and
///   only later sees real scores;
/// - row 5 masked on the second kv half: real scores first, then all-masked
///   blocks that must not disturb the accumulated state;
/// - a scattered pattern elsewhere.
fn fully_masked_rows(strategy: Strategy) {
    let (seq_q, seq_kv, head_dim, val_dim) = (16usize, 16usize, 16usize, 16usize);
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let problem = problem(
        f16_dtypes(&client),
        (seq_q, seq_kv, head_dim, val_dim),
        false,
        true,
    );

    let mut mask_values = vec![0.0f32; seq_q * seq_kv];
    for i in 0..seq_q {
        for j in 0..seq_kv {
            let masked = i < 4
                || (i == 4 && j < seq_kv / 2)
                || (i == 5 && j >= seq_kv / 2)
                || (i + j) % 7 == 0;
            mask_values[i * seq_kv + j] = masked as u32 as f32;
        }
    }

    let (query_handle, query_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Query)),
    )
    .dtype(problem.global_dtypes.query)
    .uniform(12, -1., 1.)
    .generate_with_f32_host_data();

    let (key_handle, key_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Key)),
    )
    .dtype(problem.global_dtypes.key)
    .uniform(34, -1., 1.)
    .generate_with_f32_host_data();

    let (value_handle, value_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Value)),
    )
    .dtype(problem.global_dtypes.value)
    .uniform(56, -1., 1.)
    .generate_with_f32_host_data();

    let (mask_handle, mask_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Mask)),
    )
    .dtype(problem.global_dtypes.mask)
    .custom(mask_values)
    .generate_with_bool_host_data();

    let out_handle = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Out)),
    )
    .dtype(problem.global_dtypes.out)
    .zeros()
    .generate_without_host_data();

    let problem_for_launch = problem.clone();
    let out_binding = out_handle.clone().binding();
    let outcome = launch_and_capture_outcome(&client, |c| {
        launch_ref(
            strategy.clone(),
            c,
            query_handle.clone().binding(),
            key_handle.clone().binding(),
            value_handle.clone().binding(),
            Some(mask_handle.clone().binding()),
            out_binding.clone(),
            &problem_for_launch.global_dtypes,
            problem_for_launch.options,
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual =
                HostData::from_tensor_handle(&client, out_handle.clone(), HostDataType::F32);
            for i in 0..4 {
                for d in 0..val_dim {
                    let val = actual.get_f32(&[0, 0, i, d]);
                    assert_eq!(
                        val, 0.0,
                        "fully-masked row {i} must be exactly zero, got {val} at col {d}"
                    );
                }
            }

            assert_result(
                &query_data,
                &key_data,
                &value_data,
                Some(&mask_data),
                &problem,
                &client,
                out_handle,
                AttentionElems::from_global_types(
                    &problem.global_dtypes,
                    half::f16::as_type_native_unchecked().storage_type(),
                    &problem.options.accumulator_precision,
                ),
            )
            .as_test_outcome()
            .enforce()
        }
    }
}

#[test]
fn fully_masked_rows_unit() {
    fully_masked_rows(unit_inferred())
}

#[test]
fn fully_masked_rows_blackbox() {
    fully_masked_rows(blackbox_accelerated_inferred())
}
