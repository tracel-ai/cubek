//! The SwiGLU family: `silu(gate) ⊙ up` over a fused `[gate | up]` row, in `f32` whatever the
//! rows' element.
//!
//! One dispatch reads the fused rows once, gate half and up half at a line each, and writes the
//! gated half-width result.
//!
//! This module holds the routine problem, tensors, blueprint, `launch` (the routine trait), the
//! candidate that launches a routine at a setting, holding the launches it prepared
//! ([`SwigluRoutineCandidate`]), and `routines`, one routine per scope of microkernel. The
//! microkernels, and what they read — the axes, operands, work item and partitioning — are
//! [`cubek_microkernel::swiglu`]'s. The work item is a run of one row, which a group of a cube's
//! units deals between them; nothing meets, so the routine asks nothing of the hardware and is the
//! floor of a caller that measures the routines.
//!
//! The fused rows are one operand, `[r, h, d]`: the gate at `h = 0` and the up at `h = 1`, a
//! half `d` elements apart, so the up half is reached by a stride of the one binding, never by a
//! view starting inside the buffer, and the width a line is read at is the one that stride
//! admits.
//!
//! **Traffic**, per row: `d` read from either half and `d` written. Nothing else crosses the bus,
//! which is what makes the floor a calculation.
//!
//! cubek measures nothing. A caller that does races every setting of the routine
//! ([`SwigluRoutineCandidate::list`]) on a call, keyed by [`SwigluKey`], and reads the launch each
//! prepares for a key ([`SwigluRoutinePrepared`]).

mod blueprint;
mod call;
mod candidate;
mod key;
mod launch;
mod problem;
mod routine_prepared;
mod routines;
mod tensors;

pub use blueprint::*;
pub use call::*;
pub use candidate::*;
pub use key::*;
pub use launch::*;
pub use problem::*;
pub use routine_prepared::*;
pub use routines::*;
pub use tensors::*;
