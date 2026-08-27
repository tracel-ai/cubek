//! Benchmark catalogue for `cubek-attention`.
//!
//! Gated behind the `benchmarks` cargo feature. The top-level `benchmarks`
//! crate re-exports [`Category`] from here and aggregates it with the other
//! kernels' catalogues.

mod benchmark;
mod correctness;
mod strategy;

pub use benchmark::bench;
pub use correctness::AttentionCorrectness;
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

use crate::eval::problem::{self, AttentionSpec};
use crate::forward::definition::{AttentionCost, AttentionGlobalTypes};
use crate::forward::launch::Strategy;

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
        problem::problems()
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

    /// `bench` always runs in `half::f16`, mask included.
    fn work(&self, problem: &AttentionSpec) -> Option<CategoryWork> {
        let dtype = half::f16::elem_type_native();

        let cost = AttentionCost {
            dims: problem.dims.clone(),
            masked: problem.masked,
            causal: problem.options.causal,
            types: AttentionGlobalTypes {
                query: dtype,
                key: dtype,
                value: dtype,
                mask: dtype,
                out: dtype,
            },
        };

        let (bytes_read, bytes_written) = cost.traffic();

        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(&client()),
            }),
            bytes_read,
            bytes_written,
        })
    }
}
