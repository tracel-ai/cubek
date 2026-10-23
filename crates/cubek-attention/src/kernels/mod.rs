//! Attention: `out = softmax(query · keyᵀ · scale) · value` over a float or a packed cache, at
//! any number of query positions.
//!
//! [`select`] picks how a call runs; the routines in `routines`, one folder per scope, are what
//! runs. Every routine writes the output `[b, q, h, dv]`, a token's heads adjacent, so the caller
//! merges them with a reshape.

pub mod select;

mod binding;
mod blueprint;
mod launch;
mod packed;
mod problem;
mod routines;
mod scoring;
mod tensors;
pub mod tune;

pub use binding::*;
pub use blueprint::*;
pub use launch::*;
pub use packed::*;
pub use problem::*;
pub use routines::*;
pub use scoring::*;
pub use tensors::*;
pub use tune::AttentionRoutineCandidate;

pub use cubek_microkernel::attention::AttentionVisibility;

pub use select::{AttentionKernel, AttentionKernelHeuristic, AttentionOpStrategy};

#[cfg(test)]
mod test_support;
