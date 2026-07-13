//! Kernel 3: compute `dK` and `dV`. KV-outer loop, accumulates in registers,
//! no atomics.
//!
//! Each program instance owns one KV-block and sweeps all Q-blocks. Both
//! `dK` and `dV` are accumulated independently across the sweep and written
//! once at the end.
//!
//! Implemented as two separate kernels (one for dK, one for dV) because the
//! combined kernel hits the wgpu backend's 8-storage-buffer-per-stage cap.
//! dV doesn't need `V` or `D` so its binding set is naturally smaller; dK
//! drops `dv` instead.

use cubecl::{CubeDim, Runtime, calculate_cube_count_elemwise, client::ComputeClient, prelude::*};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dv_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<E>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: StorageType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let dv_base = row_idx * val_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;

    let scale_e = E::cast_from(scale);

    let mut dv_acc = Array::new(val_dim);
    for dd in 0..val_dim {
        dv_acc[dd] = E::new(0.0_f32);
    }

    for i in 0..seq_q {
        let masked = causal && j > i;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];

            let mut dot = E::new(0.0_f32);
            for dd in 0..head_dim {
                dot += q[q_base + dd] * k[k_base + dd];
            }
            let p_ij = (dot * scale_e - lse_i).exp();

            for dd in 0..val_dim {
                dv_acc[dd] += p_ij * do_[do_base + dd];
            }
        }
    }

    for dd in 0..val_dim {
        dv[dv_base + dd] = dv_acc[dd];
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dk_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<E>,
    d: &Tensor<E>,
    dk: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: StorageType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let v_base = row_idx * val_dim;
    let dk_base = row_idx * head_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;

    let scale_e = E::cast_from(scale);

    let mut dk_acc = Array::new(head_dim);
    for dd in 0..head_dim {
        dk_acc[dd] = E::new(0.0_f32);
    }

    for i in 0..seq_q {
        let masked = causal && j > i;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];
            let d_i = d[row_idx_base + i];

            let mut dot = E::new(0.0_f32);
            for dd in 0..head_dim {
                dot += q[q_base + dd] * k[k_base + dd];
            }
            let p_ij = (dot * scale_e - lse_i).exp();

            let mut dp = E::new(0.0_f32);
            for dd in 0..val_dim {
                dp += do_[do_base + dd] * v[v_base + dd];
            }

            let s_ds = scale_e * p_ij * (dp - d_i);
            for dd in 0..head_dim {
                dk_acc[dd] += s_ds * q[q_base + dd];
            }
        }
    }

    for dd in 0..head_dim {
        dk[dk_base + dd] = dk_acc[dd];
    }
}

/// Compute `dK` and `dV` into pre-allocated tensors (overwrites, not
/// accumulates).
///
/// Inputs:
/// - `q, k, v`: `[B, H, N, d]`.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
/// - `lse`:     `[B, H, N]` fp32 — saved from forward.
/// - `d`:       `[B, H, N]` fp32 — from the prepass kernel.
///
/// Outputs:
/// - `dk`:      `[B, H, N, d]` — written cleanly.
/// - `dv`:      `[B, H, N, d]` — written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_dkdv<R: Runtime>(
    client: &ComputeClient<R>,
    q: TensorBinding<R>,
    k: TensorBinding<R>,
    v: TensorBinding<R>,
    do_: TensorBinding<R>,
    lse: TensorBinding<R>,
    d: TensorBinding<R>,
    dk: TensorBinding<R>,
    dv: TensorBinding<R>,
    _global_dtypes: &AttentionGlobalTypes,
    config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    let head_dim = q.shape[q.shape.len() - 1];
    let val_dim = v.shape[v.shape.len() - 1];

    let working_units = k.shape.iter().product::<usize>() / head_dim;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = q
        .required_address_type(dtype.size())
        .max(k.required_address_type(dtype.size()))
        .max(v.required_address_type(dtype.size()))
        .max(do_.required_address_type(dtype.size()))
        .max(lse.required_address_type(dtype.size()))
        .max(d.required_address_type(dtype.size()))
        .max(dk.required_address_type(dtype.size()))
        .max(dv.required_address_type(dtype.size()));

    flash_attention_backward_dv_kernel::launch::<R>(
        client,
        cube_count.clone(),
        cube_dim,
        address_type,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        do_.clone().into_tensor_arg(),
        lse.clone().into_tensor_arg(),
        dv.into_tensor_arg(),
        config.scale,
        head_dim,
        val_dim,
        config.causal,
        dtype,
    );

    flash_attention_backward_dk_kernel::launch::<R>(
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
        dk.into_tensor_arg(),
        config.scale,
        head_dim,
        val_dim,
        config.causal,
        dtype,
    );

    Ok(())
}
