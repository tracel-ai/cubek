//! Sliding-window acceptance tier.
//!
//! Contract: bottom-right aligned, like the causal mask. Query row `i` sees at
//! most `W` keys, the window *ending* at its own position `i + seq_kv - seq_q`,
//! so the last query row always sits on the last key. A window of 1 is
//! "attend only yourself"; a window at least `max(seq_q, seq_kv)` wide is
//! indistinguishable from plain causal masking.
//!
//! Because the window's right edge is the same one causal uses, a window
//! *subsumes* causal: `causal + window` and `window` alone see exactly the same
//! keys, which is pinned below kernel-to-kernel. The window is still plumbed
//! independently of the `causal` flag, so either can be set alone.
//!
//! Most tests here compare a windowed launch against a launch that expresses
//! the same visible set through a *materialized* mask, written from the spec
//! interval rather than from the kernel's arithmetic. That pins both edges of
//! the window, and it cannot pass if the kernel and the CPU reference agree on
//! the same off-by-one — a failure mode a reference comparison alone has.

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

fn problem(
    client: &Client,
    dims: AttentionDims,
    causal: bool,
    sliding_window: Option<u32>,
    masked: bool,
) -> AttentionProblem {
    AttentionProblem {
        dims,
        masked,
        global_dtypes: f16_dtypes(client),
        options: AttentionOptions {
            causal,
            sliding_window,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

fn mha(seq_q: usize, seq_kv: usize, head_dim: usize) -> AttentionDims {
    AttentionDims {
        batch: 2,
        num_heads: 4,
        num_kv_heads: 4,
        seq_q,
        seq_kv,
        head_dim,
        val_dim: head_dim,
    }
}

/// The visible keys, straight from the contract: the `W` keys ending at the
/// query's own position. Inclusive bounds.
fn window_bounds(i: usize, seq_q: usize, seq_kv: usize, window: usize) -> (usize, usize) {
    let right = i + seq_kv - seq_q;
    let left = (right + 1).saturating_sub(window);
    (left, right)
}

/// The *masked* predicate — `true` where the entry is dropped, matching the
/// materialized mask's contract (a stored `1` means masked out).
fn window_mask_pred(dims: AttentionDims, window: usize) -> Vec<bool> {
    let mut mask = Vec::with_capacity(dims.batch * dims.num_heads * dims.seq_q * dims.seq_kv);
    for _b in 0..dims.batch {
        for _h in 0..dims.num_heads {
            for i in 0..dims.seq_q {
                let right = i + dims.seq_kv - dims.seq_q;
                for j in 0..dims.seq_kv {
                    let visible = j + window > right && j <= right;
                    mask.push(!visible);
                }
            }
        }
    }
    mask
}

/// Deterministic data over a whole `[batch, heads, seq, dim]` tensor, distinct
/// per element so a transposed or wrongly indexed read cannot pass. `heads` is
/// passed separately because Key and Value carry `num_kv_heads`, not
/// `num_heads`.
fn wave(batch: usize, heads: usize, seq: usize, dim: usize, phase: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(batch * heads * seq * dim);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                for d in 0..dim {
                    let x = (b * 7 + h * 13 + s * 5 + d * 3) as f32 * phase;
                    data.push(x.sin() + 0.5 * x.cos());
                }
            }
        }
    }
    data
}

fn upload_f32(client: &Client, shape: [usize; 4], data: &[f32]) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f16_dtypes(client).query)
        .custom(data.to_vec())
        .generate_with_f32_host_data()
        .0
}

fn upload_bool(client: &Client, shape: [usize; 4], data: &[bool]) -> TensorHandle {
    // The mask dtype is u8 or u32, not `bool`, and `custom` carries f32 host
    // data; nonzero stores as a set entry.
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f16_dtypes(client).mask)
        .custom(data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect())
        .generate_with_bool_host_data()
        .0
}

#[allow(clippy::too_many_arguments)]
/// Launch once on explicit host data and return the output as [`HostData`].
fn run_once(
    client: &Client,
    dims: AttentionDims,
    causal: bool,
    sliding_window: Option<u32>,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<&[bool]>,
) -> HostData {
    let problem = problem(client, dims, causal, sliding_window, mask.is_some());

    let query_handle = upload_f32(client, problem.shape(AttentionIdent::Query), query);
    let key_handle = upload_f32(client, problem.shape(AttentionIdent::Key), key);
    let value_handle = upload_f32(client, problem.shape(AttentionIdent::Value), value);
    let mask_handle = mask.map(|m| upload_bool(client, problem.shape(AttentionIdent::Mask), m));

    let out_handle = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Out)),
    )
    .dtype(problem.global_dtypes.out)
    .zeros()
    .generate_without_host_data();

    let out_binding = out_handle.clone().binding();
    let mask_binding = mask_handle.map(|m| m.binding());
    let options = problem.options.clone();

    let outcome = launch_and_capture_outcome(client, &[&out_handle.handle], |c| {
        launch_ref(
            unit_inferred(),
            c,
            query_handle.binding(),
            key_handle.binding(),
            value_handle.binding(),
            mask_binding,
            out_binding,
            &problem.global_dtypes,
            options,
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

/// A windowed launch must see exactly the keys an equivalent materialized mask
/// exposes. Both edges are pinned: a window one too wide keeps a key the mask
/// rejects, one too narrow drops a key it keeps.
fn assert_window_matches_mask(dims: AttentionDims, window: usize, causal: bool) {
    let client = cubecl::test_device().client();
    let query = wave(dims.batch, dims.num_heads, dims.seq_q, dims.head_dim, 0.017);
    let key = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.head_dim,
        0.011,
    );
    let value = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.val_dim,
        0.023,
    );

    let windowed = run_once(
        &client,
        dims.clone(),
        causal,
        Some(window as u32),
        &query,
        &key,
        &value,
        None,
    );

    let mask = window_mask_pred(dims.clone(), window);
    let masked = run_once(
        &client,
        dims.clone(),
        causal,
        None,
        &query,
        &key,
        &value,
        Some(&mask),
    );

    assert_equals_approx(&windowed, &masked, 1e-2)
        .as_test_outcome()
        .enforce();
}

/// Square prefill, windows from the degenerate 1 up past the sequence length.
#[test]
fn window_matches_equivalent_materialized_mask() {
    for window in [1usize, 2, 5, 17, 64, 200] {
        assert_window_matches_mask(mha(64, 64, 32), window, true);
    }
}

/// Decode against a cache: `seq_q < seq_kv` shifts the window so it still ends
/// on the newest key, and a window wider than the whole cache masks nothing.
#[test]
fn window_matches_equivalent_materialized_mask_decode() {
    for window in [1usize, 8, 33, 96, 512] {
        assert_window_matches_mask(mha(1, 96, 32), window, true);
    }
}

/// The window's right edge is the query's own position — causal's edge — so
/// with or without the flag the visible set is identical. A window that
/// secretly depended on `causal` (or that causal could silently widen) fails
/// here.
#[test]
fn window_is_independent_of_causal() {
    let client = cubecl::test_device().client();
    let dims = mha(48, 96, 32);
    let query = wave(dims.batch, dims.num_heads, dims.seq_q, dims.head_dim, 0.017);
    let key = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.head_dim,
        0.011,
    );
    let value = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.val_dim,
        0.023,
    );

    for window in [1u32, 3, 9, 48, 96] {
        let causal = run_once(
            &client,
            dims.clone(),
            true,
            Some(window),
            &query,
            &key,
            &value,
            None,
        );
        let plain = run_once(
            &client,
            dims.clone(),
            false,
            Some(window),
            &query,
            &key,
            &value,
            None,
        );
        assert_equals_approx(&causal, &plain, 0.0)
            .as_test_outcome()
            .enforce();
    }
}

/// Windowed decode with LFM2's grouping: 32 query heads over 8 KV heads, window
/// 8 against a 96-key cache.
#[test]
fn windowed_gqa_decode() {
    assert_window_matches_mask(
        AttentionDims {
            batch: 1,
            num_heads: 32,
            num_kv_heads: 8,
            seq_q: 1,
            seq_kv: 96,
            head_dim: 32,
            val_dim: 32,
        },
        8,
        true,
    );
}

/// A window wider than the sequence must not mask a single extra entry: the
/// windowed launch has to reproduce the unmasked causal launch. This is what
/// guards the sentinel that disables the window.
#[test]
fn window_wider_than_the_sequence_is_causal() {
    let client = cubecl::test_device().client();
    let dims = mha(64, 64, 32);
    let query = wave(dims.batch, dims.num_heads, dims.seq_q, dims.head_dim, 0.017);
    let key = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.head_dim,
        0.011,
    );
    let value = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.val_dim,
        0.023,
    );

    let causal = run_once(
        &client,
        dims.clone(),
        true,
        None,
        &query,
        &key,
        &value,
        None,
    );
    let wide = run_once(&client, dims, true, Some(4096), &query, &key, &value, None);

    assert_equals_approx(&wide, &causal, 0.0)
        .as_test_outcome()
        .enforce();
}

/// The visible set, read straight out of the numbers: with zero logits the
/// softmax is uniform over whatever survives, so each output row is the *mean*
/// value over the keys that survived. A window shifted or resized by one moves
/// the mean by a half-step, far above f16 noise at these magnitudes.
#[test]
fn windowed_rows_average_exactly_the_visible_keys() {
    let client = cubecl::test_device().client();
    let (seq_q, seq_kv, head_dim) = (64usize, 64usize, 16usize);
    let window = 5usize;
    let dims = AttentionDims {
        batch: 1,
        num_heads: 1,
        num_kv_heads: 1,
        seq_q,
        seq_kv,
        head_dim,
        val_dim: head_dim,
    };

    // Zero query and key: every logit is exactly 0 regardless of position.
    let zeros = vec![0.0; dims.batch * dims.num_heads * seq_q * head_dim];
    let key = vec![0.0; dims.batch * dims.num_heads * seq_kv * head_dim];
    // Value row `j` is the constant `j`, so the mean recovers the window.
    let value: Vec<f32> = (0..dims.batch * dims.num_heads * seq_kv)
        .flat_map(|bh_j| {
            let j = bh_j % seq_kv;
            std::iter::repeat_n(j as f32, head_dim)
        })
        .collect();

    let out = run_once(
        &client,
        dims,
        true,
        Some(window as u32),
        &zeros,
        &key,
        &value,
        None,
    );

    for i in 0..seq_q {
        let (left, right) = window_bounds(i, seq_q, seq_kv, window);
        let expected = (left..=right).map(|j| j as f32).sum::<f32>() / (right - left + 1) as f32;
        for d in 0..head_dim {
            let got = out.get_f32(&[0, 0, i, d]);
            assert!(
                (got - expected).abs() < 0.1,
                "row {i} dim {d}: {got} != {expected}, window should be [{left}, {right}]"
            );
        }
    }
}

/// Reference-backed coverage: the shared launcher checks the kernel against the
/// CPU reference, which carries the same window arithmetic written out
/// separately in `cpu_reference`.
#[test]
fn windowed_launch_matches_reference() {
    let client = cubecl::test_device().client();
    let mut dims = mha(32, 32, 16);
    dims.batch = 1;

    for window in [1u32, 4, 12, 32] {
        let problem = problem(&client, dims.clone(), true, Some(window), false);
        test_launch(client.clone(), problem, unit_inferred());
    }
}

/// Window and materialized mask together: the visible set is their conjunction.
/// A bug that lets either one win outright shows up as a key that survives.
#[test]
fn window_intersects_with_materialized_mask() {
    let client = cubecl::test_device().client();
    let dims = mha(48, 48, 32);
    let window = 7usize;
    let query = wave(dims.batch, dims.num_heads, dims.seq_q, dims.head_dim, 0.017);
    let key = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.head_dim,
        0.011,
    );
    let value = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.val_dim,
        0.023,
    );
    // Also a masked predicate: `true` drops the entry.
    let random_mask: Vec<bool> = (0..dims.batch * dims.num_heads * dims.seq_q * dims.seq_kv)
        .map(|k| (k * 2654435761usize) % 100 < 40)
        .collect();

    let windowed = run_once(
        &client,
        dims.clone(),
        true,
        Some(window as u32),
        &query,
        &key,
        &value,
        Some(&random_mask),
    );

    // The kernel's should_mask is `logical || materialized`, so the equivalent
    // single mask is the OR of the two masked predicates.
    let conjunction = window_mask_pred(dims.clone(), window)
        .into_iter()
        .zip(random_mask.iter().copied())
        .map(|(w, m)| w || m)
        .collect::<Vec<_>>();

    // The conjunction can empty a row entirely, and a fully masked row's output
    // is not specified; compare against a launch carrying that conjunction as a
    // plain mask, skipping rows that keep nothing.
    let expected = run_once(
        &client,
        dims.clone(),
        false,
        None,
        &query,
        &key,
        &value,
        Some(&conjunction),
    );

    for b in 0..dims.batch {
        for h in 0..dims.num_heads {
            for i in 0..dims.seq_q {
                let base = (b * dims.num_heads + h) * dims.seq_q * dims.seq_kv + i * dims.seq_kv;
                let keeps_something = (0..dims.seq_kv).any(|j| !conjunction[base + j]);
                if !keeps_something {
                    continue;
                }
                for d in 0..dims.val_dim {
                    let got = windowed.get_f32(&[b, h, i, d]);
                    let want = expected.get_f32(&[b, h, i, d]);
                    assert!(
                        (got - want).abs() < 1e-2,
                        "row {i} dim {d}: {got} != {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn window_bounds_edge_non_dividing_stages() {
    let client = cubecl::test_device().client();
    // Make stage sizes not divide seq_kv to stress skip_start/skip_end
    let mut dims = mha(5, 11, 32);
    dims.batch = 1;
    dims.num_heads = 2;
    dims.num_kv_heads = 2;
    for window in [1u32, 3, 7] {
        let problem = problem(&client, dims.clone(), true, Some(window), false);
        test_launch(client.clone(), problem, unit_inferred());
    }
}

#[test]
fn window_bounds_decode_long_cache() {
    let client = cubecl::test_device().client();
    let dims = mha(1, 257, 16);
    // Test various windows including ones that cut into cache
    for window in [1u32, 8, 33, 256, 1000] {
        let problem = problem(&client, dims.clone(), true, Some(window), false);
        test_launch(client.clone(), problem, unit_inferred());
    }
}

#[test]
fn window_without_causal_flag() {
    let client = cubecl::test_device().client();
    let mut dims = mha(16, 24, 32);
    dims.batch = 1;
    // Window alone (no causal flag) - window's right edge is query's own position
    // which is same as causal alignment; test against mask
    let window = 5usize;
    let query = wave(dims.batch, dims.num_heads, dims.seq_q, dims.head_dim, 0.013);
    let key = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.head_dim,
        0.007,
    );
    let value = wave(
        dims.batch,
        dims.num_kv_heads,
        dims.seq_kv,
        dims.val_dim,
        0.019,
    );
    let windowed = run_once(
        &client,
        dims.clone(),
        false,
        Some(window as u32),
        &query,
        &key,
        &value,
        None,
    );
    let expected = run_once(
        &client,
        dims.clone(),
        false,
        None,
        &query,
        &key,
        &value,
        Some(&window_mask_pred(dims.clone(), window)),
    );
    for b in 0..dims.batch {
        for h in 0..dims.num_heads {
            for i in 0..dims.seq_q {
                for d in 0..dims.val_dim {
                    let got = windowed.get_f32(&[b, h, i, d]);
                    let want = expected.get_f32(&[b, h, i, d]);
                    assert!(
                        (got - want).abs() < 1e-2,
                        "mismatch at {b},{h},{i},{d}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
