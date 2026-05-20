//! Benchmark catalogue for the FlashAttention backward pass.
//!
//! Stubs only — the kernels are `todo!()` until the implementation lands.
//! Reuses [`crate::eval::benchmarks::problem`] for the
//! `(batch, heads, seq, ...)` catalogue; the per-kernel slices live in
//! [`strategy`]; the bench harness in [`benchmark`].

mod benchmark;
mod strategy;

pub use benchmark::bench;
pub use strategy::{BackwardStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::eval::problem::{self, AttentionSpec};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = AttentionSpec;
    type Strategy = BackwardStrategy;

    fn id(&self) -> &'static str {
        "attention_backward"
    }

    fn label(&self) -> &'static str {
        "Attention (backward)"
    }

    fn problems(&self) -> Vec<CatalogEntry<AttentionSpec>> {
        problem::problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<BackwardStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &BackwardStrategy,
        spec: &AttentionSpec,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, spec, num_samples)
    }
}
