//! Microkernels launched on their own, for tests: one work item, one cube, the problem exactly
//! the work item, so a test checks what a microkernel computes with no routine around it.
//!
//! What a launch needs that a microkernel does not state (the partitioning, the stages) is set here
//! to the least that runs it: one plane, one stage at a time. The operands' widths are the
//! microkernel's to state ([`Lines`]): a launch binds them one element a line, and again as wide
//! as the microkernel's accesses admit. A routine's choices are its own tests'.

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "attention-backward")]
pub mod attention_backward;
#[cfg(feature = "interpolate")]
pub mod interpolate;
#[cfg(feature = "linear-recurrence")]
pub mod linear_recurrence;
#[cfg(any(feature = "matmul", feature = "attention"))]
mod lines;
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
#[cfg(feature = "row-store")]
mod row_batch;
#[cfg(feature = "attention")]
pub mod split_keys_merge;
#[cfg(feature = "swiglu")]
pub mod swiglu;
#[cfg(any(feature = "pool", feature = "interpolate"))]
mod window;

#[cfg(any(feature = "matmul", feature = "attention"))]
pub use lines::Lines;
#[cfg(feature = "row-store")]
pub use row_batch::RowBatch;
