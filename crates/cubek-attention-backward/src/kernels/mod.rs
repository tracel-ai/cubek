//! The routines of attention's backward pass, in `routines`, and what they share: the problem,
//! the tensors, the walks a routine launches, and the autotune that picks one, behind
//! [`attention_backward`].

pub mod tune;

mod blueprint;
mod problem;
mod routine;
mod routines;
mod tensors;
mod walk;

pub use blueprint::*;
pub use problem::*;
pub use routine::*;
pub use routines::*;
pub use tensors::*;
pub use tune::attention_backward;
pub use walk::*;
