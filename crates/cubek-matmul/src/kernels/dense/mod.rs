//! The float product's seam: the one problem every matmul routine reads, how a launch reads it,
//! the tensors, blueprint and routine trait, and the routines by scope in `routines`. A packed
//! input no routine reads is decoded ahead of one by a plan step
//! ([`Decoded`](crate::kernels::plan::Decoded)).

mod binding;
mod blueprint;
mod counters;
mod cube_order;
mod inputs;
mod launch;
mod output;
mod partitioning;
mod problem;
mod routines;
mod tensors;
mod view;

pub use binding::*;
pub use blueprint::*;
pub use cube_order::*;
pub use inputs::*;
pub use launch::*;
pub use output::*;
pub use partitioning::*;
pub use problem::*;
pub use routines::*;
pub use tensors::*;
pub use view::*;
