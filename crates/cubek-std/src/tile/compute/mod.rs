//! Compute over tiles.
//!
//! - [`matmul`] — per-flavor tile matmul (execute / load / write / readers
//!   / writers).
//! - [`rowwise`] — row-wise primitives (`row_max`, `row_sum`, `exp_diff`,
//!   `rowwise_scale`) with the cross-unit plane reducer they depend on.
//! - [`elementwise`] — per-element ops (`scale_and_mask`, `fill_zero`).
//! - [`softmax`] — online softmax / per-row scale / output write.
//! - [`mask`] — `Mask` trait + `MaskLayout`.
//! - [`bounce`] (private) — cmma ↔ fragment synchronization helpers used by
//!   `softmax`.

pub mod matmul;
pub mod rowwise;

mod bounce;
mod elementwise;
mod mask;
mod softmax;

pub use mask::*;
pub use softmax::*;
