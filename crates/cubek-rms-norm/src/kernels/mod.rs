//! The RMS-norm family: `x / sqrt(mean(x², last) + eps) · weight`, the square-sum in `f32`
//! whatever the rows' element.
//!
//! One dispatch reads each row once at full vector width, holds it in registers, and writes it
//! scaled.
//!
//! This module holds the routine problem, tensors, blueprint, `launch` (the routine trait), the
//! candidates that launch a routine at a setting, each holding the launches it prepared
//! ([`RmsNormRoutineCandidate`], [`RmsNormCandidate`]), and `routines`, one routine per scope of
//! microkernel. The microkernels, one per scope, and what they read — the axes, operands, work
//! item, partitioning and the row an owner holds — are [`cubek_microkernel::rms_norm`]'s. The work
//! item is one row. A group of units owns it, summing through shared memory, which asks nothing of
//! the hardware and is the floor of a caller that measures the routines; a plane owns it, summing
//! across its lanes; or a cube owns it, its planes each summing a run and meeting once in shared
//! memory, which is what a single row of a model's width wants.
//!
//! **Traffic**, per row of `d` elements: `d` read and `d` written, and the weight's `d` read once a
//! cube, at the rows' width, so the secondary stream costs the request count the dominant one does.
//! Nothing else crosses the bus, which is what makes the floor a calculation.
//!
//! cubek measures nothing. A caller that does races every setting of the routines it lists
//! ([`RmsNormRoutineCandidate::list`]) on a call, keyed by [`RmsNormKey`], reads the launch each
//! prepares for a key ([`RmsNormRoutinePrepared`]), and identifies the winner as an
//! [`RmsNormCandidate`].

mod blueprint;
mod call;
mod candidate;
mod key;
mod launch;
mod problem;
mod routine_candidate;
mod routine_prepared;
mod routines;
mod tensors;

pub use blueprint::*;
pub use call::*;
pub use candidate::*;
pub use key::*;
pub use launch::*;
pub use problem::*;
pub use routine_candidate::*;
pub use routine_prepared::*;
pub use routines::*;
pub use tensors::*;
