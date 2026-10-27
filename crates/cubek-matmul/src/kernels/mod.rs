//! Matmul kernels built on the microkernels in [`microkernel`](cubek_microkernel::matmul): the dense
//! product's routines ([`MatmulRoutine`]), which kernel runs a product ([`select`]) and how autotune
//! measures them ([`tune`]). A routine reads each operand as it is stored ([`tensor`]).
//!
//! There is one product. The routed, M-grouped and K-grouped products ([`indexed`]) are the dense
//! product over batches whose inputs read a table, one entry a batch ([`MatmulIndex`]).

pub mod indexed;
pub mod select;
pub mod tensor;
pub mod tune;

mod dense;

pub use dense::*;
pub use select::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
pub use tune::MatmulRoutineCandidate;
