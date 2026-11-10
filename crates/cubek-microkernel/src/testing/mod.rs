//! Microkernels launched on their own, for tests: one work item, one cube, the problem exactly
//! the work item, so a test checks what a microkernel computes with no routine around it.
//!
//! What a launch needs that a microkernel does not state (the partitioning, the stages and how
//! the operands are bound) is set here to the least that runs it: one plane, one stage at a
//! time, every operand bound one element a line. A routine's choices are its own tests'.

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "interpolate")]
pub mod interpolate;
#[cfg(feature = "linear-recurrence")]
pub mod linear_recurrence;
#[cfg(feature = "matmul")]
pub mod matmul;
#[cfg(feature = "pool")]
pub mod pool;
#[cfg(feature = "random")]
pub mod random;
#[cfg(feature = "reduce")]
pub mod reduce;
#[cfg(feature = "rms-norm")]
pub mod rms_norm;
#[cfg(feature = "attention")]
pub mod split_keys_merge;
#[cfg(feature = "swiglu")]
pub mod swiglu;
#[cfg(any(feature = "pool", feature = "interpolate"))]
mod window;
