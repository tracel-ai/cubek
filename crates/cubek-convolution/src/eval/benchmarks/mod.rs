//! Benchmark catalogue for `cubek-convolution`.
//!
//! Gated behind the `benchmarks` cargo feature. The top-level `benchmarks`
//! crate re-exports [`Category`] from here and aggregates it with the other
//! kernels' catalogues.

mod benchmark;
mod category;
mod correctness;
pub mod depthwise;
pub mod direct;
mod problem;
pub mod settings;
mod strategy;

pub use benchmark::bench;
pub use category::Category;
pub use correctness::Conv2dCorrectness;
pub use problem::{Conv2dElement, Conv2dProblem, problems};
pub use strategy::strategies;
