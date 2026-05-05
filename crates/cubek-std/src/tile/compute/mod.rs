//! Compute over tiles.
//!
//! - [`matmul`] — per-flavor tile matmul (execute / load / write / readers
//!   / writers) plus the `Tile::mma` dispatch.
//! - [`rowwise`] — row-wise primitives (`row_max`, `row_sum`, `exp_diff`,
//!   `rowwise_scale`) with the cross-unit plane reducer they depend on.
//! - [`elementwise`] — per-element ops (`scale_and_mask`, `fill_zero`).
//! - [`softmax`] — online softmax / per-row scale / output write.
//! - [`copy`] — `Tile::copy_from` dispatch across tile flavors, plus the
//!   private cmma ↔ fragment synchronization helpers used by `softmax`.
//! - [`mask`] — `Mask` trait + `MaskLayout`.

pub mod matmul;
pub mod rowwise;

mod copy;
mod elementwise;
mod mask;
mod softmax;

pub use mask::*;
pub use softmax::*;
