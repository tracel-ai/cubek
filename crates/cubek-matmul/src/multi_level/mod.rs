//! Matmul routines written on the batch/global/stage/tile levels.

/// Tensor argument DSL (`MatmulArgs` and friends) shared by the multi-level launch paths.
pub mod args;
/// Components for matrix multiplication
pub mod components;
/// Blueprint, tiling scheme, and precision vocabulary specific to multi-level.
pub mod definition;
/// Contains matmul kernels
pub mod routines;
/// Shared memory stages the levels read tiles out of.
pub mod stage;
/// The tile abstraction the innermost level computes on.
pub mod tile;

mod cube_dim_resource;
mod instruction;
mod plane_flow;
mod routine;
mod select_kernel;
mod size;
#[allow(clippy::module_inception)]
mod strategy;
#[cfg(feature = "extended")]
pub mod test_only;

pub use cube_dim_resource::*;
pub use instruction::*;
pub use plane_flow::*;
pub use routine::*;
pub use select_kernel::*;
pub use size::*;
pub use strategy::Strategy;
