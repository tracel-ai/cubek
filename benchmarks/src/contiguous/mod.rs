mod benchmark;
mod problem;
mod strategy;

pub use problem::{ContiguousProblem, problems};
pub use strategy::{ContiguousStrategy, strategies};

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = ContiguousProblem;
    type Strategy = ContiguousStrategy;

    fn id(&self) -> &'static str {
        "contiguous"
    }

    fn label(&self) -> &'static str {
        "Contiguous"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<ContiguousProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ContiguousStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ContiguousStrategy,
        problem: &ContiguousProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }
}
