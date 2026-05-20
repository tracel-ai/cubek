#![allow(clippy::explicit_counter_loop, clippy::type_complexity)]

/// FlashAttention backward pass (scaffolded — kernel bodies are `todo!()`).
pub mod backward;
/// Tile/stage/global components shared by both forward and backward.
pub mod components;
/// CPU references and benchmark catalogues (forward + backward + shared
/// pieces).
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
/// FlashAttention forward pass.
pub mod forward;
