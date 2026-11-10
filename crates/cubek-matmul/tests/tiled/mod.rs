//! Tests for the routines written on the tile DSL.

mod cmma;
mod elements;
mod finish;
mod indexed;
mod integers;
mod packed;
mod packed_activation;
mod split_k;
mod storage;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
#[cfg(feature = "extended")]
mod cpu_gemm;
#[cfg(feature = "extended")]
#[cfg(feature = "extended")]
mod stride_zero;
