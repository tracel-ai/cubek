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

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

/// High-level FlashAttention backward.
///
/// Allocates the `D` prepass tensor internally, then dispatches the three
/// kernels: prepass → dQ → dK/dV. dQ and dK/dV could be launched
/// concurrently; the scaffold sequences them for simplicity until autotuning
/// lands.
///
/// Inputs:
/// - `q, k, v`: `[B, H, N, d]`.
/// - `o`:       `[B, H, N, d]` — saved from forward.
/// - `lse`:     `[B, H, N]` fp32 — saved from forward.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
///
/// Outputs (caller pre-allocates, matching the convention of `launch_ref`):
/// - `dq, dk, dv`: `[B, H, N, d]` — written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward<R: Runtime>(
    _client: &ComputeClient<R>,
    _q: TensorBinding<R>,
    _k: TensorBinding<R>,
    _v: TensorBinding<R>,
    _o: TensorBinding<R>,
    _lse: TensorBinding<R>,
    _do_: TensorBinding<R>,
    _dq: TensorBinding<R>,
    _dk: TensorBinding<R>,
    _dv: TensorBinding<R>,
    _global_dtypes: &AttentionGlobalTypes,
    _config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    todo!("flash_attention_backward orchestration not yet implemented")
}
