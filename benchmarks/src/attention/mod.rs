mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use problem::{AttentionSpec, problems};
pub use strategy::strategies;

use cubek::attention::launch::Strategy;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
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
        benchmark::bench(strategy, spec, num_samples)
    }

    #[cfg(feature = "cpu-reference")]
    fn correctness(
        &self,
    ) -> Option<&dyn crate::registry::Correctness<Problem = AttentionSpec, Strategy = Strategy>>
    {
        Some(&correctness::AttentionCorrectness)
    }
}
