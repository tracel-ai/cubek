//! Matmul routines written on the batch/global/stage/tile level tower.

/// Tensor argument DSL (`MatmulArgs` and friends) shared by the tower launch paths.
pub mod args;
/// Components for matrix multiplication
pub mod components;
/// Blueprint, tiling scheme, and precision vocabulary specific to the tower.
pub mod definition;
/// Contains matmul kernels
pub mod routines;

mod routine;
mod select_kernel;
#[allow(clippy::module_inception)]
mod strategy;
#[cfg(feature = "extended")]
pub mod test_only;

pub use routine::*;
pub use select_kernel::*;
pub use strategy::Strategy;
