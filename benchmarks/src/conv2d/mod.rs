mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::{Conv2dProblem, problems};
pub use strategy::strategies;

use cubek::convolution::Strategy;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = Conv2dProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "conv2d"
    }

    fn label(&self) -> &'static str {
        "Conv2d"
    }

    fn problems(&self) -> Vec<CatalogEntry<Conv2dProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &Conv2dProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<&dyn crate::registry::Correctness<Problem = Conv2dProblem, Strategy = Strategy>>
    {
        Some(&correctness::Conv2dCorrectness)
    }
}
