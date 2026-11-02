//! The SwiGLU family: `silu(gate) ⊙ up` over a fused `[gate | up]` row, in `f32` whatever the
//! rows' element.
//!
//! One dispatch reads the fused rows once, gate half and up half at a line each, and writes the
//! gated half-width result.
//!
//! This module holds the routine problem, tensors, blueprint, `launch` (the routine trait), `tune`
//! (the autotune, and the candidates that launch a routine at a setting, each holding the launches
//! it prepared), and `routines`, one routine per scope of microkernel. The microkernels, and what
//! they read — the axes, operands, work item and partitioning — are
//! [`cubek_microkernel::swiglu`]'s. The work item is a run of one row, which a group of a cube's
//! units deals between them; nothing meets, so the routine asks nothing of the hardware and is the
//! autotune's floor.
//!
//! The fused rows are one operand, `[r, h, d]`: the gate at `h = 0` and the up at `h = 1`, a
//! half `d` elements apart, so the up half is reached by a stride of the one binding, never by a
//! view starting inside the buffer, and the width a line is read at is the one that stride
//! admits.
//!
//! **Traffic**, per row: `d` read from either half and `d` written. Nothing else crosses the bus,
//! which is what makes the floor a calculation.

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
    SwigluCall, SwigluKey, SwigluOutput, SwigluRoutineCandidate, launch_into_tuned, launch_tuned,
};
