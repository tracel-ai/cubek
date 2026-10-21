//! Matmul routines written on the tile DSL.

pub mod storage;

#[allow(clippy::module_inception)]
mod strategy;

pub use strategy::Strategy;

#[cfg(feature = "benchmarks")]
pub mod eval;
