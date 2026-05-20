//! Kernel 1: prepass `D = rowsum(dO ⊙ O)`, shape `[B, H, N]`, fp32.
//!
//! One program instance per row (or row-block). Output dtype must be fp32 —
//! `D` is subtracted from `dP` in the softmax Jacobian and that step is the
//! most numerically sensitive part of the backward.

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::forward::definition::AttentionSetupError;

/// Compute `D = rowsum(dO ⊙ O)` into a pre-allocated fp32 tensor.
///
/// Inputs:
/// - `o`:   `[B, H, N, d]` — output of the forward pass.
/// - `do_`: `[B, H, N, d]` — upstream gradient.
///
/// Output:
/// - `d`:   `[B, H, N]` fp32 — written cleanly.
pub fn flash_attention_backward_prepass<R: Runtime>(
    _client: &ComputeClient<R>,
    _o: TensorBinding<R>,
    _do_: TensorBinding<R>,
    _d: TensorBinding<R>,
) -> Result<(), AttentionSetupError> {
    todo!("backward prepass not yet implemented")
}
