//! Benchmark / correctness registry traits shared across kernel crates.
//!
//! Each kernel crate exposes a `pub mod benchmarks` (gated behind its
//! `benchmarks` cargo feature) that defines a [`Category`] over the kernel's
//! own problem and strategy types. The top-level `benchmarks` crate then
//! collects those into a single `all()` slice for harnesses (Cargo benches)
//! and for the tuner-runner.

mod base;
mod benchmark_category;
mod category;
mod correctness;

pub use base::*;
pub use benchmark_category::*;
pub use category::*;
pub use correctness::*;
