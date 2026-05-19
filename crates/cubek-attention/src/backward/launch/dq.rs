//! Kernel 2: compute `dQ`. Q-outer loop, accumulates in registers, no atomics.
//!
//! Each program instance owns one Q-block and sweeps all KV-blocks. `S` and
//! `dP` are recomputed inside the loop — never materialized to HBM.

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

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
    _client: &ComputeClient<R>,
    _q: TensorBinding<R>,
    _k: TensorBinding<R>,
    _v: TensorBinding<R>,
    _do_: TensorBinding<R>,
    _lse: TensorBinding<R>,
    _d: TensorBinding<R>,
    _dq: TensorBinding<R>,
    _global_dtypes: &AttentionGlobalTypes,
    _config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    todo!("backward dQ kernel not yet implemented")
}
