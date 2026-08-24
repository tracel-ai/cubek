//! Tests for the routines written on the tile DSL.

mod cmma;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
#[cfg(feature = "extended")]
mod cpu_gemm;
#[cfg(feature = "extended")]
mod stride_zero;
