//! Microkernels launched on their own, for tests: one work item, one cube, the problem exactly
//! the work item, so a test checks what a microkernel computes with no routine around it.
//!
//! What a launch needs that a microkernel does not state (the partitioning, the stages and how
//! the operands are bound) is set here to the least that runs it: one plane, one stage at a
//! time, every operand bound one element a line. A routine's choices are its own tests'.

#[cfg(feature = "matmul")]
pub mod matmul;
