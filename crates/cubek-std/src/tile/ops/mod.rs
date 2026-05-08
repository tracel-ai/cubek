//! Cross-variant tile dispatchers and shared protocols. Per-variant tile
//! data, configs, and compute helpers (matmul-execute, load/write/zero-init,
//! readers/writers) live in [`crate::tile::variants`]; the dispatchers here are
//! thin matches on `&self.kind` that delegate to per-variant methods.
//!
//! - [`matmul`] — `Tile::mma` dispatch.
//! - [`rowwise`] — `Tile::row_max` / `row_sum` / `exp_diff` / `rowwise_scale`
//!   dispatch + the cross-plane reducer used by `WhiteboxFragment`.
//! - [`elementwise`] — `Tile::scale_and_mask` / `fill_zero` dispatch.
//! - [`softmax`] — online softmax + per-row scale + output write.
//! - [`copy`] — `Tile::copy_from` and `Tile::init_zero` dispatch.
//! - [`mask`] — `Mask` trait + `MaskLayout`.
//!
//! # Dispatch contract
//!
//! Every `Tile::<op>` defined here is a thin match on `&self.kind` (or
//! `&mut self.kind`) that delegates to a method on the variant's data
//! struct in [`crate::tile::variants`]. The body of each arm should be one
//! call — no inlined loops, no per-variant special cases.
//!
//! Adding a new variant to [`crate::tile::TileKind`] therefore costs:
//!
//! 1. A new module in `tile/data/` (or extend an existing one) with the
//!    per-op methods: rowwise (`row_max`, `row_sum`, `exp_diff`,
//!    `rowwise_scale`), elementwise (`scale_and_mask`, `fill_zero`),
//!    `init_zero`, plus matmul (`mma` method + the `*_execute` /
//!    `*_load_from_shared` / `*_write_to_shared` helpers for
//!    matmul-capable variants).
//! 2. One arm per dispatcher (`matmul.rs`, `rowwise/base.rs`,
//!    `elementwise.rs`, `copy.rs`, optionally `mask.rs` and `softmax.rs`).
//!
//! No central match grows beyond an arm; no other variant's code needs
//! to change.

mod copy;
mod elementwise;
mod mask;
mod matmul;
mod rowwise;
mod softmax;

pub use softmax::*;
