//! Benchmark catalogue for `cubek-interpolate`.
mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::InterpolateCorrectness;
pub use problem::problems;
pub use strategy::{InterpolateBenchmarkStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::definition::InterpolateProblem;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateBenchmarkStrategy;

    fn id(&self) -> &'static str {
        "interpolate"
    }

    fn label(&self) -> &'static str {
        "Interpolate"
    }

    fn problems(&self) -> Vec<CatalogEntry<InterpolateProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &InterpolateBenchmarkStrategy,
        problem: &InterpolateProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<
        &dyn cubek_test_utils::Correctness<
            Problem = InterpolateProblem,
            Strategy = InterpolateBenchmarkStrategy,
        >,
    > {
        Some(&InterpolateCorrectness)
    }
}
