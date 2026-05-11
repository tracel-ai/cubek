mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{ContiguousProblem, problems};
pub use strategy::{ContiguousStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
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
        bench(strategy, problem, num_samples)
    }
}
