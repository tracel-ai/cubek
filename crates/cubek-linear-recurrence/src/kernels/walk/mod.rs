//! The walk of either update rule: the recurrence a layer of a linear recurrence family carries per
//! value head, walked through a sequence's positions. A decode step is a walk of one position.
//!
//! It is one dispatch that reads the queries, keys, values and the state once and writes the
//! output and the kept states once. Its states are tensors it owns, or the slots of a store that a
//! table places its sequences at: it reads a sequence's state there at the start and writes the
//! state after its last position back, whatever the number of positions.
//!
//! The problem, tensors, blueprint, `launch` (its routine trait) and [`tune`] live here, and
//! `routines`, one per scope of microkernel; the microkernels and what they read are
//! [`cubek_microkernel::linear_recurrence::walk`]'s.
//!
//! **Traffic**, per sequence and value head over `t` positions: the queries and keys of its key
//! head, `2 · t · key_features`, shared by the value heads grouped under it; `t · value_features`
//! values and outputs; two rates a position, and under the additive rule one skip a value head; the
//! state read once and each kept state written once, `key_features · value_features` each.

mod blueprint;
mod launch;
mod problem;
mod routines;
mod tensors;
pub mod tune;

pub use blueprint::*;
pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use tensors::*;
pub use tune::{launch_tuned, serves};
