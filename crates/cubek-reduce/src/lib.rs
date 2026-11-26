//! Reductions on CubeCL's tile stack: a tensor of any rank reduced along one dim, by a sum, a
//! product, a mean, an extremum, a flag or a top-k, its values and their indices.
//!
//! [`kernels::Reduce`] is the entry point. It is built from a [`kernels::ReduceOpStrategy`] that
//! says how the kernel is picked (the rule, or a given [`kernels::ReduceKernel`]),
//! and reduces with [`execute`](kernels::Reduce::execute) or, writing the indices beside the
//! values, [`execute_with_indices`](kernels::Reduce::execute_with_indices).
//!
//! A caller that binds the input itself, as a fused reduction reads it, states its tensors as
//! [`kernels::ReduceTensors`] and launches a [`kernels::ReduceCandidate`]. cubek measures nothing:
//! a caller that does races every [`kernels::ReduceCandidate`] the device lists, keyed by
//! [`kernels::ReduceKey`], and launches the fastest. Under the `benchmarks` feature,
//! `eval::benchmarks::ReduceCategory` is its entry in the benchmark catalogue.

pub mod kernels;

mod error;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;

pub use cubek_microkernel::reduce::ReduceOperation;
pub use error::*;
