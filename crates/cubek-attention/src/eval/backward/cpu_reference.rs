//! CPU reference for the FlashAttention backward pass.
//!
//! Ground truth for the GPU backward kernels — naive `O(B·H·N²·d)`
//! materialization of `S`, `P`, `dP`, `dS`. Slow on bench-scale problems by
//! design.

#![allow(clippy::needless_range_loop)]

use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, StridedLayout};

use crate::forward::definition::AttentionProblem;

/// Outputs of [`flash_attention_backward_reference`].
pub struct FlashAttentionBackwardResult {
    /// `[B, H, seq_q, head_dim]`.
    pub dq: HostData,
    /// `[B, H, seq_kv, head_dim]`.
    pub dk: HostData,
    /// `[B, H, seq_kv, val_dim]`.
    pub dv: HostData,
}

/// Outputs of [`flash_attention_backward_reference_debug`]: the three grads
/// plus the materialized intermediates.
pub struct FlashAttentionBackwardDebug {
    pub dq: HostData,
    pub dk: HostData,
    pub dv: HostData,
    /// Per-row `lse = m + log(ℓ)`, shape `[B, H, seq_q]`, fp32.
    pub lse: HostData,
    /// Per-row `D = rowsum(dO ⊙ O)`, shape `[B, H, seq_q]`, fp32.
    pub d: HostData,
    /// Forward output `O = P V`, shape `[B, H, seq_q, val_dim]`.
    pub o: HostData,
    /// `P`, shape `[B, H, seq_q, seq_kv]`, fp32.
    pub p: HostData,
    /// `dP = dO V^T`, shape `[B, H, seq_q, seq_kv]`, fp32.
    pub dp: HostData,
    /// `dS = P ⊙ (dP - D)`, shape `[B, H, seq_q, seq_kv]`, fp32.
    pub ds: HostData,
}

/// Ground-truth FlashAttention backward.
///
/// Materializes `S`, `P`, `dP`, `dS` explicitly — slow on bench-scale
/// problems, intended only as a correctness oracle for the GPU kernels.
///
/// `lse` and `d_prepass` are taken as inputs so per-kernel tests can swap
/// in either CPU- or GPU-computed values. For the variant that computes
/// every intermediate itself, see
/// [`flash_attention_backward_reference_debug`].
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_reference(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    do_: &HostData,
    lse: &HostData,
    d_prepass: &HostData,
    problem: &AttentionProblem,
) -> FlashAttentionBackwardResult {
    let debug = compute_backward_inner(query, key, value, do_, Some(lse), Some(d_prepass), problem);
    FlashAttentionBackwardResult {
        dq: debug.dq,
        dk: debug.dk,
        dv: debug.dv,
    }
}

/// Like [`flash_attention_backward_reference`] but also returns `lse`, `D`,
/// `O`, `P`, `dP`, `dS`. Used by unit tests that check intermediates in
/// isolation.
pub fn flash_attention_backward_reference_debug(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    do_: &HostData,
    problem: &AttentionProblem,
) -> FlashAttentionBackwardDebug {
    compute_backward_inner(query, key, value, do_, None, None, problem)
}

#[allow(clippy::too_many_arguments)]
fn compute_backward_inner(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    do_: &HostData,
    lse_in: Option<&HostData>,
    d_in: Option<&HostData>,
    problem: &AttentionProblem,
) -> FlashAttentionBackwardDebug {
    let batch = problem.dims.batch;
    let num_heads = problem.dims.num_heads;
    let seq_q = problem.dims.seq_q;
    let seq_kv = problem.dims.seq_kv;
    let head_dim = problem.dims.head_dim;
    let val_dim = problem.dims.val_dim;
    let causal = problem.options.causal;
    let scale = (head_dim as f32).sqrt().recip();

    let q_shape = Shape::new([batch, num_heads, seq_q, head_dim]);
    let k_shape = Shape::new([batch, num_heads, seq_kv, head_dim]);
    let v_shape = Shape::new([batch, num_heads, seq_kv, val_dim]);
    let o_shape = Shape::new([batch, num_heads, seq_q, val_dim]);
    let row_shape = Shape::new([batch, num_heads, seq_q]);
    let attn_shape = Shape::new([batch, num_heads, seq_q, seq_kv]);

    let mut dq = vec![0f32; batch * num_heads * seq_q * head_dim];
    let mut dk = vec![0f32; batch * num_heads * seq_kv * head_dim];
    let mut dv = vec![0f32; batch * num_heads * seq_kv * val_dim];

    let mut p_buf = vec![0f32; batch * num_heads * seq_q * seq_kv];
    let mut dp_buf = vec![0f32; batch * num_heads * seq_q * seq_kv];
    let mut ds_buf = vec![0f32; batch * num_heads * seq_q * seq_kv];
    let mut lse_buf = vec![0f32; batch * num_heads * seq_q];
    let mut d_buf = vec![0f32; batch * num_heads * seq_q];
    let mut o_buf = vec![0f32; batch * num_heads * seq_q * val_dim];

    for b in 0..batch {
        for h in 0..num_heads {
            // Per-row online softmax: produces P, O, lse for this (b, h).
            for i in 0..seq_q {
                let mut s_row = vec![f32::NEG_INFINITY; seq_kv];
                let mut m_i = f32::NEG_INFINITY;
                for j in 0..seq_kv {
                    if causal && j > i {
                        continue;
                    }
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        dot += query.get_f32(&[b, h, i, d]) * key.get_f32(&[b, h, j, d]);
                    }
                    let s_ij = dot * scale;
                    s_row[j] = s_ij;
                    if s_ij > m_i {
                        m_i = s_ij;
                    }
                }

                let mut sum_exp = 0f32;
                for j in 0..seq_kv {
                    if s_row[j].is_finite() {
                        sum_exp += (s_row[j] - m_i).exp();
                    }
                }

                let lse_i = if sum_exp > 0.0 {
                    m_i + sum_exp.ln()
                } else {
                    f32::NEG_INFINITY
                };
                lse_buf[((b * num_heads) + h) * seq_q + i] = lse_i;

                for j in 0..seq_kv {
                    let p_ij = if s_row[j].is_finite() {
                        (s_row[j] - lse_i).exp()
                    } else {
                        0.0
                    };
                    let p_idx = ((b * num_heads + h) * seq_q + i) * seq_kv + j;
                    p_buf[p_idx] = p_ij;
                }
                for d in 0..val_dim {
                    let mut acc = 0f32;
                    for j in 0..seq_kv {
                        let p_idx = ((b * num_heads + h) * seq_q + i) * seq_kv + j;
                        acc += p_buf[p_idx] * value.get_f32(&[b, h, j, d]);
                    }
                    o_buf[((b * num_heads + h) * seq_q + i) * val_dim + d] = acc;
                }

                let mut d_i = 0f32;
                for d in 0..val_dim {
                    let o_id = o_buf[((b * num_heads + h) * seq_q + i) * val_dim + d];
                    d_i += do_.get_f32(&[b, h, i, d]) * o_id;
                }
                d_buf[((b * num_heads) + h) * seq_q + i] = d_i;
            }

            if let Some(lse_in) = lse_in {
                for i in 0..seq_q {
                    lse_buf[((b * num_heads) + h) * seq_q + i] = lse_in.get_f32(&[b, h, i]);
                }
            }
            if let Some(d_in) = d_in {
                for i in 0..seq_q {
                    d_buf[((b * num_heads) + h) * seq_q + i] = d_in.get_f32(&[b, h, i]);
                }
            }
            if lse_in.is_some() {
                for i in 0..seq_q {
                    let lse_i = lse_buf[((b * num_heads) + h) * seq_q + i];
                    for j in 0..seq_kv {
                        let masked = causal && j > i;
                        let p_ij = if masked {
                            0.0
                        } else {
                            let mut dot = 0f32;
                            for d in 0..head_dim {
                                dot += query.get_f32(&[b, h, i, d]) * key.get_f32(&[b, h, j, d]);
                            }
                            (dot * scale - lse_i).exp()
                        };
                        let p_idx = ((b * num_heads + h) * seq_q + i) * seq_kv + j;
                        p_buf[p_idx] = p_ij;
                    }
                }
            }

            for i in 0..seq_q {
                let d_i = d_buf[((b * num_heads) + h) * seq_q + i];
                for j in 0..seq_kv {
                    let p_idx = ((b * num_heads + h) * seq_q + i) * seq_kv + j;
                    let mut dp_ij = 0f32;
                    for d in 0..val_dim {
                        dp_ij += do_.get_f32(&[b, h, i, d]) * value.get_f32(&[b, h, j, d]);
                    }
                    dp_buf[p_idx] = dp_ij;

                    let p_ij = p_buf[p_idx];
                    let ds_ij = p_ij * (dp_ij - d_i);
                    ds_buf[p_idx] = ds_ij;

                    for d in 0..val_dim {
                        let dv_idx = ((b * num_heads + h) * seq_kv + j) * val_dim + d;
                        dv[dv_idx] += p_ij * do_.get_f32(&[b, h, i, d]);
                    }

                    let s_ds = scale * ds_ij;
                    for d in 0..head_dim {
                        let dq_idx = ((b * num_heads + h) * seq_q + i) * head_dim + d;
                        dq[dq_idx] += s_ds * key.get_f32(&[b, h, j, d]);
                        let dk_idx = ((b * num_heads + h) * seq_kv + j) * head_dim + d;
                        dk[dk_idx] += s_ds * query.get_f32(&[b, h, i, d]);
                    }
                }
            }
        }
    }

    let q_strides = StridedLayout::RowMajor.compute_strides(&q_shape);
    let k_strides = StridedLayout::RowMajor.compute_strides(&k_shape);
    let v_strides = StridedLayout::RowMajor.compute_strides(&v_shape);
    let o_strides = StridedLayout::RowMajor.compute_strides(&o_shape);
    let row_strides = StridedLayout::RowMajor.compute_strides(&row_shape);
    let attn_strides = StridedLayout::RowMajor.compute_strides(&attn_shape);

    FlashAttentionBackwardDebug {
        dq: HostData {
            data: HostDataVec::F32(dq),
            shape: q_shape.clone(),
            strides: q_strides,
        },
        dk: HostData {
            data: HostDataVec::F32(dk),
            shape: k_shape.clone(),
            strides: k_strides,
        },
        dv: HostData {
            data: HostDataVec::F32(dv),
            shape: v_shape.clone(),
            strides: v_strides,
        },
        lse: HostData {
            data: HostDataVec::F32(lse_buf),
            shape: row_shape.clone(),
            strides: row_strides.clone(),
        },
        d: HostData {
            data: HostDataVec::F32(d_buf),
            shape: row_shape,
            strides: row_strides,
        },
        o: HostData {
            data: HostDataVec::F32(o_buf),
            shape: o_shape,
            strides: o_strides,
        },
        p: HostData {
            data: HostDataVec::F32(p_buf),
            shape: attn_shape.clone(),
            strides: attn_strides.clone(),
        },
        dp: HostData {
            data: HostDataVec::F32(dp_buf),
            shape: attn_shape.clone(),
            strides: attn_strides.clone(),
        },
        ds: HostData {
            data: HostDataVec::F32(ds_buf),
            shape: attn_shape,
            strides: attn_strides,
        },
    }
}
