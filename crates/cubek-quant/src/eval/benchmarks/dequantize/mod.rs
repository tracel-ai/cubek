//! Elementwise dequantize: the legacy kernels against the tile engine.
//!
//! Every problem runs under both strategies in one process, so each pair of numbers is an
//! interleaved A/B on the same device state. The two-level problem runs tile-only: the
//! legacy kernels serve one scale level.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{DequantizeProblem, problems};
pub use strategy::{DequantizePath, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = DequantizeProblem;
    type Strategy = DequantizePath;

    fn id(&self) -> &'static str {
        "dequantize"
    }

    fn label(&self) -> &'static str {
        "Quant: elementwise dequantize"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<DequantizeProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<DequantizePath>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &DequantizePath,
        problem: &DequantizeProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
