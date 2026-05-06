mod benchmark;
mod problem;
mod strategy;

pub use problem::{MemcpyAsyncProblem, problems};
pub use strategy::{CopyStrategyEnum, strategies};

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = MemcpyAsyncProblem;
    type Strategy = CopyStrategyEnum;

    fn id(&self) -> &'static str {
        "memcpy_async"
    }

    fn label(&self) -> &'static str {
        "Memcpy (async)"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<MemcpyAsyncProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<CopyStrategyEnum>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &CopyStrategyEnum,
        problem: &MemcpyAsyncProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }
}
