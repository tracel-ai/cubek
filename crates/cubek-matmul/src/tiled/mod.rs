//! Matmul routines written on the tile DSL.

pub mod cmma;
pub mod cpu_gemm;
pub mod quant_gemv;
pub mod storage;

mod matrix;
mod operands;
#[allow(clippy::module_inception)]
mod strategy;

pub(crate) use matrix::*;
pub(crate) use operands::*;
pub use strategy::Strategy;

#[cfg(feature = "benchmarks")]
pub mod eval;
