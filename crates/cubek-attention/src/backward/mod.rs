//! FlashAttention backward pass.
//!
//! Three-kernel pipeline (same shape as FlashAttention-3):
//! 1. [`flash_attention_backward_prepass`] — compute `D = rowsum(dO ⊙ O)`.
//! 2. [`flash_attention_backward_dq`]      — compute `dQ` with a Q-outer
//!    loop, no atomics.
//! 3. [`flash_attention_backward_dkdv`]    — compute `dK` and `dV` with a
//!    KV-outer loop, no atomics.
//!
//! Kernels 2 and 3 are independent and may run concurrently once the
//! prepass is done. The high-level [`flash_attention_backward`] entry point
//! hides the orchestration and is what a framework autograd hook should
//! call.

pub mod definition;
pub mod launch;
pub mod routines;

pub use definition::{BackwardConfig, TileShape};
pub use launch::{
    flash_attention_backward, flash_attention_backward_dkdv, flash_attention_backward_dq,
    flash_attention_backward_prepass,
};
