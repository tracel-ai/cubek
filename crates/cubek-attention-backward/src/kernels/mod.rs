//! The routines of attention's backward pass, one kernel each, in `routines`; the strategies that
//! launch them in order, in `strategies`; and what they share: the problem, the tensors, the walk
//! a routine launches, and what a caller that measures the strategies races: each at each of its
//! settings ([`AttentionBackwardCandidate`]), keyed by [`AttentionBackwardKey`].

mod binding;
mod candidate;
mod gradients;
mod key;
mod problem;
mod routines;
mod stages;
mod strategies;
mod tensors;
mod walk;

pub(crate) use binding::*;
pub use candidate::*;
pub use gradients::*;
pub use key::*;
pub use problem::*;
pub use routines::*;
pub(crate) use stages::*;
pub use strategies::*;
pub use tensors::*;
pub use walk::*;
