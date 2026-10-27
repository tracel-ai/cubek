#![allow(clippy::explicit_counter_loop, clippy::type_complexity)]

/// Tile/stage/global components of the forward pass.
pub mod components;
/// CPU references and benchmark catalogues of the forward pass.
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
/// FlashAttention forward pass.
pub mod forward;
#[cfg(feature = "tiled")]
pub mod kernels;
