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
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::eval::problem::{self, AttentionSpec};
use crate::forward::definition::AttentionIdent;
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

    /// Flash-attention flop count: one multiply-add per (batch, head, query
    /// position, key position) for each of `QK^T` and the `softmax(QK^T) V`
    /// matmul, so `head_dim + val_dim` rather than `2 * head_dim`. `bench` always
    /// runs in `half::f16`; the mask, when present, is approximated at that size
    /// too since its actual dtype needs a client `work` doesn't have.
    fn work(&self, problem: &AttentionSpec) -> Option<CategoryWork> {
        let dtype = half::f16::elem_type_native();
        let elem_size = dtype.size();
        let dims = &problem.dims;

        let numel = |ident| dims.shape(ident).iter().product::<usize>();
        let mut bytes_read = (numel(AttentionIdent::Query)
            + numel(AttentionIdent::Key)
            + numel(AttentionIdent::Value))
            * elem_size;
        if problem.masked {
            bytes_read += numel(AttentionIdent::Mask) * elem_size;
        }

        Some(CategoryWork {
            compute_ops: 2
                * dims.batch
                * dims.num_heads
                * dims.seq_q
                * dims.seq_kv
                * (dims.head_dim + dims.val_dim),
            dtype,
            bytes_read,
            bytes_written: numel(AttentionIdent::Out) * elem_size,
        })
    }
}
