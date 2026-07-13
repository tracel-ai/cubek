//! LSE oracle tier: host-only checks of the CPU reference's `lse = m + ln(l)`
//! output against a direct (non-online) logsumexp. No kernel emits lse yet;
//! these pin the convention the tile port's epilogue will be tested against:
//! natural log, and exactly `-inf` on fully-masked rows (matching the
//! backward reference).

use cubecl::frontend::CubePrimitive;
use cubecl::ir::AddressType;
use cubecl::zspace::Shape;
use cubek_attention::eval::forward::cpu_reference::flash_attention_v2_reference_with_lse;
use cubek_attention::forward::definition::{
    AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
};
use cubek_test_utils::{HostData, HostDataVec, StridedLayout};

fn host_f32(shape: [usize; 4], f: impl Fn(usize) -> f32) -> HostData {
    let shape = Shape::new(shape);
    let n = shape.iter().product();
    HostData {
        data: HostDataVec::F32((0..n).map(f).collect()),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    }
}

fn host_bool(shape: [usize; 4], f: impl Fn(usize) -> bool) -> HostData {
    let shape = Shape::new(shape);
    let n = shape.iter().product();
    HostData {
        data: HostDataVec::Bool((0..n).map(f).collect()),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    }
}

/// Deterministic pseudo-random floats in [-1, 1] with no RNG dependency.
fn wobble(i: usize, salt: usize) -> f32 {
    (((i * 2654435761 + salt * 40503) % 2048) as f32 / 1024.) - 1.
}

fn problem(
    (batch, num_heads, seq_q, seq_kv, head_dim, val_dim): (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    causal: bool,
    masked: bool,
) -> AttentionProblem {
    AttentionProblem {
        dims: AttentionDims {
            batch,
            num_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked,
        global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
            f32::as_type_native_unchecked(),
            u8::as_type_native_unchecked().storage_type(),
        ),
        options: AttentionOptions {
            causal,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

/// Direct per-row logsumexp: one max pass, one sum pass, no online recurrence.
/// Fully-masked rows give `-inf`.
fn direct_lse(
    query: &HostData,
    key: &HostData,
    mask: Option<&HostData>,
    problem: &AttentionProblem,
    (b, h, i): (usize, usize, usize),
) -> f32 {
    let dims = &problem.dims;
    let scale = (dims.head_dim as f32).sqrt().recip();

    let mut scores = Vec::with_capacity(dims.seq_kv);
    for j in 0..dims.seq_kv {
        if problem.options.causal && j > i {
            continue;
        }
        if let Some(mask) = mask
            && mask.get_bool(&[b, h, i, j])
        {
            continue;
        }
        let mut dot = 0.;
        for d in 0..dims.head_dim {
            dot += query.get_f32(&[b, h, i, d]) * key.get_f32(&[b, h, j, d]);
        }
        scores.push(dot * scale);
    }

    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if scores.is_empty() {
        return f32::NEG_INFINITY;
    }
    m + scores.iter().map(|s| (s - m).exp()).sum::<f32>().ln()
}

/// Crafted mask with fully-masked rows plus scattered holes; the online lse
/// must match the direct logsumexp everywhere, and be exactly -inf on the
/// fully-masked rows.
#[test]
fn lse_matches_direct_logsumexp_masked() {
    let dims = (1usize, 2usize, 16usize, 24usize, 8usize, 8usize);
    let (_, num_heads, seq_q, seq_kv, head_dim, _) = dims;
    let problem = problem(dims, false, true);

    let query = host_f32([1, num_heads, seq_q, head_dim], |i| wobble(i, 1));
    let key = host_f32([1, num_heads, seq_kv, head_dim], |i| wobble(i, 2));
    let value = host_f32([1, num_heads, seq_kv, head_dim], |i| wobble(i, 3));
    // Head 0: rows 0..3 fully masked, scattered elsewhere. Head 1: unmasked.
    let mask = host_bool([1, num_heads, seq_q, seq_kv], |lin| {
        let h = lin / (seq_q * seq_kv);
        let i = (lin / seq_kv) % seq_q;
        let j = lin % seq_kv;
        h == 0 && (i < 3 || (i + j) % 7 == 0)
    });

    let (out, lse) =
        flash_attention_v2_reference_with_lse(&query, &key, &value, Some(&mask), &problem, None);

    for h in 0..num_heads {
        for i in 0..seq_q {
            let got = lse.get_f32(&[0, h, i]);
            let expected = direct_lse(&query, &key, Some(&mask), &problem, (0, h, i));
            if h == 0 && i < 3 {
                assert_eq!(got, f32::NEG_INFINITY, "fully-masked row ({h},{i})");
                for d in 0..dims.5 {
                    assert_eq!(out.get_f32(&[0, h, i, d]), 0., "row ({h},{i}) col {d}");
                }
            } else {
                assert!(
                    (got - expected).abs() <= 1e-5 * expected.abs().max(1.),
                    "lse mismatch at ({h},{i}): online {got} vs direct {expected}"
                );
            }
        }
    }
}

#[test]
fn lse_matches_direct_logsumexp_causal() {
    let dims = (1usize, 1usize, 16usize, 16usize, 8usize, 8usize);
    let (_, _, seq_q, seq_kv, head_dim, _) = dims;
    let problem = problem(dims, true, false);

    let query = host_f32([1, 1, seq_q, head_dim], |i| wobble(i, 4));
    let key = host_f32([1, 1, seq_kv, head_dim], |i| wobble(i, 5));
    let value = host_f32([1, 1, seq_kv, head_dim], |i| wobble(i, 6));

    let (_, lse) =
        flash_attention_v2_reference_with_lse(&query, &key, &value, None, &problem, None);

    for i in 0..seq_q {
        let got = lse.get_f32(&[0, 0, i]);
        let expected = direct_lse(&query, &key, None, &problem, (0, 0, i));
        assert!(
            (got - expected).abs() <= 1e-5 * expected.abs().max(1.),
            "lse mismatch at row {i}: online {got} vs direct {expected}"
        );
    }
}
