//! The RMS-norm family: `x / sqrt(mean(x², last) + eps) · weight`, the square-sum in `f32`
//! whatever the rows' element.
//!
//! Composed of tensor operations, a norm runs as ~8 element-wise dispatches — the cast up, the
//! square, the mean, the epsilon, the root and the cast down, then the scale — none of which can
//! fuse where norms sit between custom projections, which are fusion boundaries. Here it is one
//! dispatch that reads each row once at full vector width, holds it in registers, and writes it
//! scaled.
//!
//! **The microkernel norm** lives here: its routine problem, tensors, blueprint, `launch` (its
//! routine trait) and `tune` (the autotune, and the candidates that launch a routine at a setting,
//! each holding the launches it prepared), and `routines`, one routine per scope of microkernel. The
//! microkernels themselves, one per scope, and what they read — the axes, operands, work item,
//! partitioning and the row an owner holds — are [`cubek_microkernel::rms_norm`]'s. The work item
//! is one row. A group of units owns it, summing through shared memory, which asks nothing of the
//! hardware and is the autotune's floor; a plane owns it, summing across its lanes; or a cube owns it,
//! its planes each summing a run and meeting once in shared memory, which is what a decode step's
//! single row of the model's width wants. The [`tune`] autotune races every routine's settings
//! per key.
//!
//! **Traffic**, per row of `d` elements: `d` read and `d` written, and the weight's `d` read once a
//! cube, at the rows' width, so the secondary stream costs the request count the dominant one does.
//! Nothing else crosses the bus, which is what makes the floor a calculation.

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
pub use tune::{
    RmsNormCall, RmsNormCandidate, RmsNormKey, RmsNormOutput, RmsNormRoutineCandidate,
    RmsNormRoutineSettings, fastest_rms_norm_routine, launch_into_tuned, launch_tuned,
};
