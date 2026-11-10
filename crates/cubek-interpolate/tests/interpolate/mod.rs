//! The forward and backward interpolations launched as kernels, against the crate's host
//! reference.

mod backward;
#[cfg(feature = "benchmarks")]
mod bench_catalog;
mod forward;
