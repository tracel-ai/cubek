//! Matrix multiplication kernels.
//!
//! Two architectures live here, and the crate branches between them at its root:
//! `tiled` holds routines written on the tile DSL, `multi_level` holds routines
//! written on the batch/global/stage/tile levels. Everything outside those two
//! modules is vocabulary both of them speak. Each is behind a feature of its own
//! name, both on by default, and either compiles without the other.

/// The problem, its element types, and the errors either architecture can raise.
pub mod definition;
/// The public launch entry point.
pub mod launch;
/// The contract a routine implements, and what it reads about the device.
pub mod routine;
/// The user-facing [`Strategy`](strategy::Strategy) enum: the branch point.
pub mod strategy;
/// Autotune key.
pub mod tune_key;

#[cfg(feature = "multi-level")]
pub mod multi_level;
#[cfg(feature = "tiled")]
pub mod tiled;

#[cfg(not(any(feature = "tiled", feature = "multi-level")))]
compile_error!("cubek-matmul needs at least one of the `tiled` and `multi-level` features");

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
