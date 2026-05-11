//! Benchmark catalogue for `cubek-attention`.
//!
//! Gated behind the `benchmarks` cargo feature. The top-level `benchmarks`
//! crate re-exports [`Category`] from here and aggregates it with the other
//! kernels' catalogues.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::AttentionCorrectness;
pub use problem::{AttentionSpec, build_problem, problems};
pub use strategy::strategies;

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::launch::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = AttentionSpec;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "attention"
    }

    fn label(&self) -> &'static str {
        "Attention"
    }

    fn problems(&self) -> Vec<CatalogEntry<AttentionSpec>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        spec: &AttentionSpec,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, spec, num_samples)
    }

    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = AttentionSpec, Strategy = Strategy>>
    {
        Some(&AttentionCorrectness)
    }
}
