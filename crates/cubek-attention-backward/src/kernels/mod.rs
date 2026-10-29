//! The routines of attention's backward pass, in `routines`, and what they share: the problem,
//! the tensors, the walks a routine launches, and the autotune that picks one, behind
//! [`attention_backward`].

mod binding;
mod blueprint;
mod gradients;
mod problem;
mod routine;
mod routines;
mod stages;
mod tensors;
mod tune;
mod walk;

pub(crate) use binding::*;
pub use blueprint::*;
pub use gradients::*;
pub use problem::*;
pub use routine::*;
pub use routines::*;
pub(crate) use stages::*;
pub use tensors::*;
pub use tune::attention_backward;
pub use walk::*;
