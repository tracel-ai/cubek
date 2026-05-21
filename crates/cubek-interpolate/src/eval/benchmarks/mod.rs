//! Benchmark catalogue for `cubek-interpolate`.
mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::InterpolateCorrectness;
pub use problem::problems;
pub use strategy::strategies;

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::{definition::InterpolateProblem, launch::InterpolateStrategy};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateStrategy;

    fn id(&self) -> &'static str {
        "interpolate"
    }

    fn label(&self) -> &'static str {
        "Interpolate"
    }

    fn problems(&self) -> Vec<CatalogEntry<InterpolateProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<InterpolateStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &InterpolateStrategy,
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
            Strategy = InterpolateStrategy,
        >,
    > {
        Some(&InterpolateCorrectness)
    }
}
