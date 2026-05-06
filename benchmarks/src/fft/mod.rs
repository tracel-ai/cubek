mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::{FftProblem, problems};
pub use strategy::{FftStrategy, strategies};

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
    type Problem = FftProblem;
    type Strategy = FftStrategy;

    fn id(&self) -> &'static str {
        "fft"
    }

    fn label(&self) -> &'static str {
        "FFT"
    }

    fn problems(&self) -> Vec<CatalogEntry<FftProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<FftStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &FftStrategy,
        problem: &FftProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        benchmark::bench(strategy, problem, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<&dyn crate::registry::Correctness<Problem = FftProblem, Strategy = FftStrategy>>
    {
        Some(&correctness::FftCorrectness)
    }
}
