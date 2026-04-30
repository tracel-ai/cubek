//! Benchmark registry for cubek.
//!
//! Each category exposes a list of strategies, a list of problems, and a
//! `run(strategy_id, problem_id, samples)` entry point.

pub mod attention;
pub mod registry;

pub use registry::{ItemDescriptor, RunSamples};
