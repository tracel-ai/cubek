//! Attention: `out = softmax(query · keyᵀ · scale) · value` over a float or a packed cache, at
//! any number of query positions.
//!
//! [`select`] picks how a call runs; the routines in `routines`, one folder per scope, are what
//! runs, each one kernel. A routine writes the output `[b, q, h, dv]`, a token's heads adjacent,
//! so the caller merges them with a reshape. A partial routine writes only the partials of each
//! run of the keys ([`PartialAttentionRoutine`]), and the split-keys merge after it the output
//! ([`SplitKeysAttention`]).

pub mod plan;
pub mod select;

mod binding;
mod blueprint;
mod conversion;
mod kv_source;
mod launch;
mod packed;
mod partitioning;
mod problem;
mod queries;
mod routines;
mod scoring;
mod split_keys;
mod tensors;
pub mod tune;

pub use binding::*;
pub use blueprint::*;
pub use kv_source::*;
pub use launch::*;
pub use packed::*;
pub use partitioning::*;
pub use problem::*;
pub use queries::*;
pub use routines::*;
pub use scoring::*;
pub use split_keys::*;
pub use tensors::*;
pub use tune::AttentionRoutineCandidate;

pub use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};

pub use select::{AttentionKernel, AttentionKernelHeuristic, AttentionOpStrategy};

#[cfg(test)]
mod test_support;
