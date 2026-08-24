//! Matrix multiplication kernels.
//!
//! Two families live here, and the crate branches between them at its root:
//! [`tiled`] holds routines written on the tile DSL, [`multi_level`] holds routines
//! written on the batch/global/stage/tile level tower. Everything outside those two
//! modules is vocabulary both of them speak.

/// The problem, its element types, and the errors either family can raise.
pub mod definition;
/// The public launch entry point.
pub mod launch;
/// The contract a routine implements, and what it reads about the device.
pub mod routine;
/// The user-facing [`Strategy`](strategy::Strategy) enum: the branch point.
pub mod strategy;
/// Autotune key.
pub mod tune_key;

pub mod multi_level;
pub mod tiled;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
