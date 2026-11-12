//! The `attention_backward` category: [`problems`] by [`strategies`], each row the two launches of
//! one backward pass in `f16`.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{AttentionBackwardSpec, problems};
pub use strategy::{AttentionBackwardStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = AttentionBackwardSpec;
    type Strategy = AttentionBackwardStrategy;

    fn id(&self) -> &'static str {
        "attention_backward"
    }

    fn label(&self) -> &'static str {
        "Attention backward"
    }

    fn problems(&self) -> Vec<CatalogEntry<AttentionBackwardSpec>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<AttentionBackwardStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &AttentionBackwardStrategy,
        spec: &AttentionBackwardSpec,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, spec, num_samples)
    }
}
