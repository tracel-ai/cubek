//! The tile DSL routines' launch strategy ([`Strategy`]), the storage relayout their tiled operands
//! are laid in ([`storage`]) and their benchmarks. The routines themselves are
//! [`kernels`](crate::kernels).

pub mod storage;

#[allow(clippy::module_inception)]
mod strategy;

pub use strategy::Strategy;

#[cfg(feature = "benchmarks")]
pub mod eval;
