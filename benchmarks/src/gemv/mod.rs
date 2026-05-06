mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::{GemvProblem, ProblemKind, problems};
pub use strategy::strategies;

use cubek::matmul::launch::Strategy;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = GemvProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "gemv"
    }

    fn label(&self) -> &'static str {
        "GEMV"
    }

    fn problems(&self) -> Vec<CatalogEntry<GemvProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &GemvProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<&dyn crate::registry::Correctness<Problem = GemvProblem, Strategy = Strategy>> {
        Some(&correctness::GemvCorrectness)
    }
}
