//! Benchmark catalogue for `cubek-convolution`.
//!
//! Gated behind the `benchmarks` cargo feature. The top-level `benchmarks`
//! crate re-exports [`Category`] from here and aggregates it with the other
//! kernels' catalogues.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::Conv2dCorrectness;
pub use problem::{Conv2dProblem, problems};
pub use strategy::strategies;

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
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
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = Conv2dProblem, Strategy = Strategy>>
    {
        Some(&Conv2dCorrectness)
    }
}
