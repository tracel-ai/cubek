//! The float product's seam: the one problem every matmul routine reads, how a launch reads it,
//! the tensors, blueprint and routine trait, and the routines by scope in `routines`.

mod blueprint;
mod counters;
mod cube_order;
mod inputs;
mod launch;
mod output;
mod problem;
mod routines;
mod tensors;
mod view;

pub use blueprint::*;
pub use cube_order::*;
pub use inputs::*;
pub use launch::*;
pub use output::*;
pub use problem::*;
pub use routines::*;
pub use tensors::*;
pub use view::*;
