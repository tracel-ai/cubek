//! Tests for the routines written on the tile DSL.

mod cmma;
mod quant_gemv;
mod storage;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
#[cfg(feature = "extended")]
mod cpu_gemm;
#[cfg(feature = "benchmarks")]
mod selector_probe;
#[cfg(feature = "extended")]
#[cfg(feature = "extended")]
mod stride_zero;
