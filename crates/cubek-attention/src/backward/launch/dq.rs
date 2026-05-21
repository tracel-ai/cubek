//! Kernel 2: compute `dQ`. Q-outer loop, accumulates in registers, no atomics.
//!
//! Each program instance owns one Q-block and sweeps all KV-blocks. `S` and
//! `dP` are recomputed inside the loop — never materialized to HBM.

use cubecl::{CubeDim, Runtime, calculate_cube_count_elemwise, client::ComputeClient, prelude::*};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dq_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<E>,
    d: &Tensor<E>,
    dq: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: StorageType,
) {
    let row_idx = ABSOLUTE_POS;
    if row_idx >= lse.len() {
        terminate!();
    }

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let i = row_idx % seq_q;
    let bh = row_idx / seq_q;

    let q_base = row_idx * head_dim;
    let do_base = row_idx * val_dim;
    let dq_base = row_idx * head_dim;
    let k_row_base = bh * seq_kv * head_dim;
    let v_row_base = bh * seq_kv * val_dim;

    let lse_i = lse[row_idx];
    let d_i = d[row_idx];
    let scale_e = E::cast_from(scale);

    let mut dq_acc = Array::<E>::new(head_dim);
    for dd in 0..head_dim {
        dq_acc[dd] = E::new(0.0);
    }

    for j in 0..seq_kv {
        let masked = causal && j > i;
        if !masked {
            let k_base = k_row_base + j * head_dim;
            let v_base = v_row_base + j * val_dim;

            let mut dot = E::new(0.0);
            for dd in 0..head_dim {
                dot += q[q_base + dd] * k[k_base + dd];
            }
            let s_ij = dot * scale_e;
            let p_ij = (s_ij - lse_i).exp();

            let mut dp = E::new(0.0);
            for dd in 0..val_dim {
                dp += do_[do_base + dd] * v[v_base + dd];
            }

            let s_ds = scale_e * p_ij * (dp - d_i);

            for dd in 0..head_dim {
                dq_acc[dd] += s_ds * k[k_base + dd];
            }
        }
    }

    for dd in 0..head_dim {
        dq[dq_base + dd] = dq_acc[dd];
    }
}

/// Compute `dQ` into a pre-allocated tensor (overwrites, not accumulates).
///
/// Inputs:
/// - `q, k, v`: `[B, H, N, d]`.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
/// - `lse`:     `[B, H, N]` fp32 — saved from forward.
/// - `d`:       `[B, H, N]` fp32 — from the prepass kernel.
///
/// Output:
/// - `dq`:      `[B, H, N, d]` — written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_dq<R: Runtime>(
    client: &ComputeClient<R>,
    q: TensorBinding<R>,
    k: TensorBinding<R>,
    v: TensorBinding<R>,
    do_: TensorBinding<R>,
    lse: TensorBinding<R>,
    d: TensorBinding<R>,
    dq: TensorBinding<R>,
    _global_dtypes: &AttentionGlobalTypes,
    config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    let head_dim = q.shape[q.shape.len() - 1];
    let val_dim = v.shape[v.shape.len() - 1];

    let working_units = lse.shape.iter().product::<usize>();
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = q
        .required_address_type(dtype.size())
        .max(k.required_address_type(dtype.size()))
        .max(v.required_address_type(dtype.size()))
        .max(do_.required_address_type(dtype.size()))
        .max(lse.required_address_type(dtype.size()))
        .max(d.required_address_type(dtype.size()))
        .max(dq.required_address_type(dtype.size()));

    flash_attention_backward_dq_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        q.into_tensor_arg(),
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        do_.into_tensor_arg(),
        lse.into_tensor_arg(),
        d.into_tensor_arg(),
        dq.into_tensor_arg(),
        config.scale,
        head_dim,
        val_dim,
        config.causal,
        dtype,
    );

    Ok(())
}
