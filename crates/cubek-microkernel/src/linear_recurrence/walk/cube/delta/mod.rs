//! The cube scope's delta rule: the microkernel that decays the state, recalls, writes the delta
//! and reads.

mod base;
mod microkernel;

pub use base::*;
pub use microkernel::*;
