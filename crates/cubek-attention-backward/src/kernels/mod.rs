//! The routines of attention's backward pass, one kernel each, in `routines`; the strategies that
//! launch them in order, in `strategies`; and what they share: the problem, the tensors, the walk
//! a routine launches, and the autotune that races the strategies, behind [`attention_backward`].

mod binding;
mod gradients;
mod problem;
mod routines;
mod stages;
mod strategies;
mod tensors;
mod tune;
mod walk;

pub(crate) use binding::*;
pub use gradients::*;
pub use problem::*;
pub use routines::*;
pub(crate) use stages::*;
pub use strategies::*;
pub use tensors::*;
pub use tune::attention_backward;
pub use walk::*;
