mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::{ReduceProblem, problems};
pub use strategy::strategies;

use cubek::reduce::launch::ReduceStrategy;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = ReduceProblem;
    type Strategy = ReduceStrategy;

    fn id(&self) -> &'static str {
        "reduce"
    }

    fn label(&self) -> &'static str {
        "Reduce"
    }

    fn problems(&self) -> Vec<CatalogEntry<ReduceProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ReduceStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<&dyn crate::registry::Correctness<Problem = ReduceProblem, Strategy = ReduceStrategy>>
    {
        Some(&correctness::ReduceCorrectness)
    }
}
