//! Reductions on CubeCL's tile stack: a tensor of any rank reduced along one dim, by a sum, a
//! product, a mean, an extremum, a flag or a top-k, its values and their indices.
//!
//! [`kernels::reduce`] and [`kernels::reduce_with_indices`] are the entry points. Each takes a
//! [`kernels::ReduceOpStrategy`] that says how the kernel is picked: the autotune, the rule, or a
//! given [`kernels::ReduceKernel`].

pub mod kernels;

mod error;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;

pub use cubek_microkernel::reduce::ReduceOperation;
pub use error::*;
