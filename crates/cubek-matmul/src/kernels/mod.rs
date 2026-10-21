//! Matmul kernels built on the microkernels in [`microkernel`](crate::microkernel): the dense
//! product's routines ([`MatmulRoutine`]), which kernel runs a product ([`select`]) and how a table
//! measures them ([`tune`]). A routine reads each operand as it is stored ([`tensor`]).

pub mod select;
pub mod tensor;
pub mod tune;

mod binding;
mod dense;

pub use binding::*;
pub use dense::*;
pub use select::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
pub use tune::MatmulRoutineCandidate;
