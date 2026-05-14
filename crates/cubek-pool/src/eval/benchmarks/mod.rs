//! Benchmark catalogue for `cubek-pool`.
mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::PoolCorrectness;
pub use problem::problems;
pub use strategy::{PoolStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::definition::PoolProblem;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = PoolProblem;
    type Strategy = PoolStrategy;

    fn id(&self) -> &'static str {
        "pool"
    }

    fn label(&self) -> &'static str {
        "Pool"
    }

    fn problems(&self) -> Vec<CatalogEntry<PoolProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<PoolStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &PoolStrategy,
        problem: &PoolProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = PoolProblem, Strategy = PoolStrategy>>
    {
        Some(&PoolCorrectness)
    }
}
