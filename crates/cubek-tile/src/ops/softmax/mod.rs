//! The softmax reading of a [`Tile`](crate::Tile): one online-softmax step
//! along the score axis absent from the state's space (label-driven, like
//! matmul's contraction).
//!
//! Deliberately leaf-scoped: softmax runs on already-lowered tiles, inside a
//! verb that owns the walk: attention's forward interleaves the step with
//! the value mma and rescales its accumulator by the returned correction; the
//! backward's row ops join here later. No self-lowering: if a standalone
//! softmax verb ever needs to walk levels and normalize, its schedules land
//! then, on a real client.

mod leaf;
mod rowwise;
mod state;

pub use state::*;
