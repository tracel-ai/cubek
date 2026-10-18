//! Grouped-query acceptance tier, pinned at the problem level (packed KV
//! tensors in, per-query-head output out, checked against the CPU reference).
//!
//! Contract: Q, Out and the mask keep one row per *query* head; K and V are
//! packed to `num_kv_heads` and are read in place — the kernel never expands
//! them. Query heads `[g * k, (g + 1) * k)` attend KV head `g`.
//!
//! The CPU reference spells the group mapping out itself rather than sharing
//! `AttentionDims::kv_head`, so `grouping_matches_expanded_mha` is what pins
//! the two against each other: it runs the GQA kernel on the packed tensors and
//! the (already trusted) MHA kernel on the same data broadcast to
//! `num_heads`, and demands identical output.

use crate::attention::forward::launcher::test_launch;
use cubecl::{
    client::Client, ir::AddressType, prelude::Scalar, std::tensor::TensorHandle, zspace::Shape,
};
use cubek_attention::{
    forward::definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionIdent,
        AttentionOptions, AttentionProblem,
    },
    forward::launch::{BlueprintStrategy, Strategy, launch_ref},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, assert_equals_approx,
    launch_and_capture_outcome,
};

fn f16_dtypes(client: &Client) -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_float_dtype(
        half::f16::elem_type_native(),
        AttentionGlobalTypes::mask_dtype(client),
    )
}

fn unit_inferred() -> Strategy {
    Strategy::Unit(BlueprintStrategy::Inferred(()))
}

fn gqa_problem(
    global_dtypes: AttentionGlobalTypes,
    (batch, num_heads, num_kv_heads): (usize, usize, usize),
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
) -> AttentionProblem {
    AttentionProblem {
        dims: AttentionDims {
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

/// Deterministic per-element data, so the packed and expanded views of the same
/// KV tensor are byte-identical: element `(b, h, s, d)` of a `[batch, heads,
/// seq, dim]` tensor is `wave(b, h, s, d)` regardless of how many heads the
/// tensor is meant to have.
fn wave(b: usize, h: usize, s: usize, d: usize) -> f32 {
    let x = (b * 7 + h * 13 + s * 5 + d * 3) as f32;
    (x * 0.017).sin() + 0.5 * (x * 0.0031).cos()
}

fn fill(shape: [usize; 4], head_of: impl Fn(usize) -> usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(shape.iter().product());
    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for s in 0..shape[2] {
                for d in 0..shape[3] {
                    v.push(wave(b, head_of(h), s, d));
                }
            }
        }
    }
    v
}

/// MQA: one KV head behind every query head. The strongest form of the mapping —
/// every K/V read must come from the single packed head.
#[test]
fn mqa_4_query_heads_1_kv_head() {
    let client = cubecl::test_device().client();
    test_launch(
        client.clone(),
        gqa_problem(f16_dtypes(&client), (1, 4, 1), (128, 128, 64, 64)),
        unit_inferred(),
    )
}

/// Group of 4, the ratio LFM2 uses (32 query heads over 8 KV heads).
#[test]
fn gqa_group_of_four() {
    let client = cubecl::test_device().client();
    test_launch(
        client.clone(),
        gqa_problem(f16_dtypes(&client), (2, 8, 2), (128, 128, 64, 64)),
        unit_inferred(),
    )
}

/// Batch and heads must stay independent under GQA: a bug that mixes the two
/// axes still passes every single-batch case.
#[test]
fn gqa_batched() {
    let client = cubecl::test_device().client();
    test_launch(
        client.clone(),
        gqa_problem(f16_dtypes(&client), (3, 4, 2), (64, 64, 32, 32)),
        unit_inferred(),
    )
}

/// The LFM2 attention geometry: 32 query heads over 8 KV heads, head_dim 64,
/// causal. Sequence lengths are kept small because the CPU reference is O(batch
/// * heads * seq_q * seq_kv * dim).
#[test]
fn lfm2_geometry_causal() {
    let client = cubecl::test_device().client();
    let mut problem = gqa_problem(f16_dtypes(&client), (1, 32, 8), (64, 64, 64, 64));
    problem.options.causal = true;

    test_launch(client.clone(), problem, unit_inferred())
}

/// A decode step against a cache: `seq_q < seq_kv` puts the causal diagonal at
/// the bottom right, and every query head must still fold down to the right KV
/// head across the cache.
#[test]
fn gqa_decode_against_cache() {
    let client = cubecl::test_device().client();
    let mut problem = gqa_problem(f16_dtypes(&client), (2, 8, 2), (1, 96, 64, 64));
    problem.options.causal = true;

    test_launch(client.clone(), problem, unit_inferred())
}

/// Grouping is a pure read-side fold: the GQA kernel on the packed KV tensors
/// must reproduce the MHA kernel on the same data expanded to `num_heads`.
/// Checked against the kernel rather than the reference, so a mapping bug
/// mirrored in both cannot pass.
#[test]
fn grouping_matches_expanded_mha() {
    let client = cubecl::test_device().client();
    let global_dtypes = f16_dtypes(&client);
    let (batch, num_heads, num_kv_heads) = (2usize, 8usize, 2usize);
    let (seq_q, seq_kv, head_dim, val_dim) = (64usize, 64usize, 32usize, 32usize);
    let group = num_heads / num_kv_heads;

    let query = fill([batch, num_heads, seq_q, head_dim], |h| h);
    // Same `wave` values, indexed by the head the reader will see: the packed
    // view keeps one entry per KV head, the expanded view repeats each of them
    // `group` times contiguously.
    let key = fill([batch, num_kv_heads, seq_kv, head_dim], |h| h);
    let value = fill([batch, num_kv_heads, seq_kv, val_dim], |h| h);
    let key_expanded = fill([batch, num_heads, seq_kv, head_dim], |h| h / group);
    let value_expanded = fill([batch, num_heads, seq_kv, val_dim], |h| h / group);

    let gqa = run_once(
        &client,
        global_dtypes.clone(),
        (batch, num_heads, num_kv_heads),
        (seq_q, seq_kv, head_dim, val_dim),
        &query,
        &key,
        &value,
    );
    let mha = run_once(
        &client,
        global_dtypes,
        (batch, num_heads, num_heads),
        (seq_q, seq_kv, head_dim, val_dim),
        &query,
        &key_expanded,
        &value_expanded,
    );

    // Expanded K/V hold exactly `group` copies of each packed head, so the two
    // launches are the same computation; only the head arithmetic differs.
    assert_equals_approx(&gqa, &mha, 0.0)
        .as_test_outcome()
        .enforce();
}

/// Launch once on explicit host data and return the output as [`HostData`].
///
/// Returns `None` when the strategy refuses to compile, so the caller reports
/// it through the same `CompileError` policy as [`test_launch`].
fn run_once(
    client: &Client,
    global_dtypes: AttentionGlobalTypes,
    (batch, num_heads, num_kv_heads): (usize, usize, usize),
    (seq_q, seq_kv, head_dim, val_dim): (usize, usize, usize, usize),
    query: &[f32],
    key: &[f32],
    value: &[f32],
) -> HostData {
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes: global_dtypes.clone(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let (query_handle, _) = upload(client, problem.shape(AttentionIdent::Query), query);
    let (key_handle, _) = upload(client, problem.shape(AttentionIdent::Key), key);
    let (value_handle, _) = upload(client, problem.shape(AttentionIdent::Value), value);
    let out_handle = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Out)),
    )
    .dtype(problem.global_dtypes.out)
    .zeros()
    .generate_without_host_data();

    let out_binding = out_handle.clone().binding();
    let outcome = launch_and_capture_outcome(client, &[&out_handle.handle], |c| {
        launch_ref(
            unit_inferred(),
            c,
            query_handle.binding(),
            key_handle.binding(),
            value_handle.binding(),
            None,
            out_binding,
            &global_dtypes,
            AttentionOptions {
                causal: false,
                accumulator_precision: AccumulatorPrecision::default(),
            },
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => panic!("{e}"),
        ExecutionOutcome::Executed => {
            HostData::from_tensor_handle(client, out_handle, HostDataType::F32)
        }
    }
}

fn upload(client: &Client, shape: [usize; 4], data: &[f32]) -> (TensorHandle, HostData) {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f16_dtypes(client).query)
        .custom(data.to_vec())
        .generate_with_f32_host_data()
}
