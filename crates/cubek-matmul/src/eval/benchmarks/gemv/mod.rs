mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::GemvCorrectness;
pub use problem::{GemvProblem, ProblemKind, problems};
pub use strategy::strategies;

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::launch::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
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
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = GemvProblem, Strategy = Strategy>>
    {
        Some(&GemvCorrectness)
    }
}
