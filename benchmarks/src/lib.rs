//! Benchmark registry for cubek.
//!
//! Each category exposes a list of strategies, a list of problems, and a
//! `run(strategy_id, problem_id, samples)` entry point. This is consumed both
//! by `cargo bench <category>` (the thin shims under `benches/`) and by the
//! external `tuner` tool which orchestrates cross-version comparisons.

pub mod attention;
pub mod registry;

pub use registry::{ItemDescriptor, RunSamples};
