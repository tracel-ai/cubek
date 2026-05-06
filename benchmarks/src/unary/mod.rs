mod benchmark;
mod problem;
mod strategy;

pub use problem::{UnaryProblem, problems};
pub use strategy::{UnaryStrategy, strategies};

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = UnaryProblem;
    type Strategy = UnaryStrategy;

    fn id(&self) -> &'static str {
        "unary"
    }

    fn label(&self) -> &'static str {
        "Unary"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<UnaryProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<UnaryStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &UnaryStrategy,
        problem: &UnaryProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }
}
