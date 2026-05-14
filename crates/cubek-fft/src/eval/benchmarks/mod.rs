//! Benchmark catalogue for `cubek-fft`.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::FftCorrectness;
pub use problem::{FftProblem, problems};
pub use strategy::{FftStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
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
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = FftProblem, Strategy = FftStrategy>>
    {
        Some(&FftCorrectness)
    }
}
