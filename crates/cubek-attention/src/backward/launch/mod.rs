//! High-level entry point and per-kernel launchers for the FlashAttention
//! backward pass.
//!
//! - [`flash_attention_backward`]: full orchestration (prepass → dQ → dK/dV).
//! - [`flash_attention_backward_prepass`]: kernel 1 only.
//! - [`flash_attention_backward_dq`]: kernel 2 only.
//! - [`flash_attention_backward_dkdv`]: kernel 3 only.

mod dkdv;
mod dq;
mod prepass;

pub use dkdv::flash_attention_backward_dkdv;
pub use dq::flash_attention_backward_dq;
pub use prepass::flash_attention_backward_prepass;

use cubecl::{client::Client, prelude::*, std::tensor::TensorHandle};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

/// The single float dtype shared by q/k/v/o/dO and the gradients. The kernels
/// are generic over it (f32, f16, bf16) and accumulate in fp32; mixed dtypes
/// are rejected rather than silently reinterpreted.
fn io_dtype(global_dtypes: &AttentionGlobalTypes) -> Result<ElemType, AttentionSetupError> {
    let dtype = global_dtypes.query;
    let uniform = [global_dtypes.key, global_dtypes.value, global_dtypes.out]
        .iter()
        .all(|t| *t == dtype);
    if !uniform || !matches!(dtype, ElemType::Float(_)) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "attention backward needs one shared float dtype for q/k/v/out, got {global_dtypes:?}"
        ))));
    }
    Ok(dtype)
}

/// High-level FlashAttention backward.
///
/// Allocates the `D` prepass tensor internally, then dispatches the three
/// kernels: prepass → dQ → dK/dV. dQ and dK/dV could be launched
/// concurrently; the scaffold sequences them for simplicity until autotuning
/// lands.
///
/// Inputs (q/k/v/o/dO/dQ/dK/dV share one float dtype, `global_dtypes.query`;
/// all reductions accumulate in fp32):
/// - `q, k, v`: `[B, H, N, d]`.
/// - `o`:       `[B, H, N, d]`: saved from forward.
/// - `lse`:     `[B, H, N]` fp32 (always, whatever the q/k/v dtype): saved from forward.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
///
/// Outputs (caller pre-allocates, matching the convention of `launch_ref`):
/// - `dq, dk, dv`: `[B, H, N, d]`: written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward(
    client: &Client,
    q: TensorBinding,
    k: TensorBinding,
    v: TensorBinding,
    o: TensorBinding,
    lse: TensorBinding,
    do_: TensorBinding,
    dq: TensorBinding,
    dk: TensorBinding,
    dv: TensorBinding,
    global_dtypes: &AttentionGlobalTypes,
    config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    let f32_dtype = f32::elem_type_native();

    // D = rowsum(dO ⊙ O), shape [B, H, N_q], always fp32.
    let d_shape = lse.shape.clone();
    let d_elems: usize = d_shape.iter().product();
    let d_handle =
        TensorHandle::new_contiguous(d_shape, client.empty(d_elems * f32_dtype.size()), f32_dtype);

    flash_attention_backward_prepass(
        client,
        o,
        do_.clone(),
        d_handle.clone().binding(),
        global_dtypes,
    )?;

    flash_attention_backward_dq(
        client,
        q.clone(),
        k.clone(),
        v.clone(),
        do_.clone(),
        lse.clone(),
        d_handle.clone().binding(),
        dq,
        global_dtypes,
        config.clone(),
    )?;

    flash_attention_backward_dkdv(
        client,
        q,
        k,
        v,
        do_,
        lse,
        d_handle.binding(),
        dk,
        dv,
        global_dtypes,
        config,
    )?;

    Ok(())
}
