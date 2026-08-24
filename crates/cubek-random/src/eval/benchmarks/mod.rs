mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{Distribution, RandomProblem, problems};
pub use strategy::{RandomStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = RandomProblem;
    type Strategy = RandomStrategy;

    fn id(&self) -> &'static str {
        "random"
    }

    fn label(&self) -> &'static str {
        "Random"
    }

    fn problems(&self) -> Vec<CatalogEntry<RandomProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<RandomStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &RandomStrategy,
        problem: &RandomProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
