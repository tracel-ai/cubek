//! Kernel 3: compute `dK` and `dV`. KV-outer loop, accumulates in registers,
//! no atomics.
//!
//! Each program instance owns one KV-block and sweeps all Q-blocks. Both
//! `dK` and `dV` are accumulated independently across the sweep and written
//! once at the end.

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

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
    _client: &ComputeClient<R>,
    _q: TensorBinding<R>,
    _k: TensorBinding<R>,
    _v: TensorBinding<R>,
    _do_: TensorBinding<R>,
    _lse: TensorBinding<R>,
    _d: TensorBinding<R>,
    _dk: TensorBinding<R>,
    _dv: TensorBinding<R>,
    _global_dtypes: &AttentionGlobalTypes,
    _config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    todo!("backward dK/dV kernel not yet implemented")
}
