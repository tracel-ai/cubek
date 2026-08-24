//! Matmul routines written on the tile DSL.

pub mod cmma;
pub mod cpu_gemm;

#[allow(clippy::module_inception)]
mod strategy;

pub use strategy::Strategy;
