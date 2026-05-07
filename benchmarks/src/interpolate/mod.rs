mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::problems;
pub use strategy::{InterpolateStrategy, strategies};

use cubek::interpolate::definition::InterpolateProblem;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
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
        benchmark::bench(strategy, problem, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<
        &dyn crate::registry::Correctness<
            Problem = InterpolateProblem,
            Strategy = InterpolateStrategy,
        >,
    > {
        Some(&correctness::InterpolateCorrectness)
    }
}
