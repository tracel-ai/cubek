//! The softmax reading of a [`Tile`](crate::Tile): one online-softmax step
//! along the score axis absent from the state's space (label-driven, like
//! matmul's contraction).
//!
//! Deliberately leaf-scoped: softmax runs on already-lowered tiles, inside a verb that owns the
//! walk (attention's forward interleaves the step with the value mma and rescales its accumulator
//! by the returned correction). No self-lowering: a standalone softmax verb's schedules land later.
//!
//! Nothing but the step. The row plumbing a caller wraps it in is
//! [`rows`](super::rows) and the split ending is attention's; either one
//! landing here is the module drifting back into a dumping ground.

mod leaf;
pub mod logsumexp;
mod planewise;
mod rowwise;
mod state;

pub use state::*;
