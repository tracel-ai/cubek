//! Tests for the routines written on the tile DSL, a folder a scope as `kernels` lays them out.

mod base;
mod cube;
mod finish;
mod hypercube;
mod indexed;
mod packed;
mod plane;
mod storage;
mod strategy;
mod unit;

#[cfg(feature = "benchmarks")]
mod bench_catalog;

pub(crate) use base::*;
