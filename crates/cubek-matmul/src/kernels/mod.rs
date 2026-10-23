//! Matmul kernels built on the microkernels in [`microkernel`](cubek_microkernel::matmul): the dense
//! product's routines ([`MatmulRoutine`]), which kernel runs a product ([`select`]) and how autotune
//! measures them ([`tune`]). A routine reads each operand as it is stored ([`tensor`]).
//!
//! The routed, M-grouped and K-grouped products ([`routed`], [`m_grouped`], [`k_grouped`]) hold no
//! routine and no kernel: each is the dense product over batches whose inputs read a table, one
//! entry a batch ([`MatmulIndex`]), its rows laid out by [`grouped_rows`] and its stack by
//! [`stack`].

pub mod grouped_rows;
pub mod k_grouped;
pub mod m_grouped;
pub mod routed;
pub mod select;
pub mod stack;
pub mod tensor;
pub mod tune;

mod binding;
mod dense;

pub use binding::*;
pub use dense::*;
pub use select::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
pub use tune::MatmulRoutineCandidate;
