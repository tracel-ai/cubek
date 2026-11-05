//! Tests for the routines written on the tile DSL.

mod cmma;
mod elements;
mod indexed;
mod integers;
mod packed;
mod packed_activation;
mod packed_fragment;
mod rows_along_n_bench;
mod storage;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
#[cfg(feature = "extended")]
mod cpu_gemm;
#[cfg(feature = "extended")]
#[cfg(feature = "extended")]
mod stride_zero;
