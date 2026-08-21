//! Benchmark catalogue for `cubek-reduce`.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::ReduceCorrectness;
pub use problem::{ReduceBenchKind, ReduceProblem, problems};
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::ReduceStrategy;
use crate::components::instructions::ReduceOperationConfig;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = ReduceProblem;
    type Strategy = ReduceStrategy;

    fn id(&self) -> &'static str {
        "reduce"
    }

    fn label(&self) -> &'static str {
        "Reduce"
    }

    fn problems(&self) -> Vec<CatalogEntry<ReduceProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ReduceStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<
        &dyn cubek_test_utils::Correctness<Problem = ReduceProblem, Strategy = ReduceStrategy>,
    > {
        Some(&ReduceCorrectness)
    }

    /// No honest compute count (a reduce is a mix of adds and comparisons depending
    /// on `config`), so this declares reads and writes only. `TwoLaunch` reads the
    /// input twice and writes both halves separately; `Fused` reads it once but still
    /// writes both halves; `Single` does neither extra.
    fn work(&self, problem: &ReduceProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();

        let input_elems: usize = problem.shape.iter().product();
        let reduce_len = match problem.config {
            ReduceOperationConfig::ArgTopK(len) | ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        let output_elems = (input_elems / problem.shape[problem.axis]) * reduce_len;

        let (read_passes, write_halves) = match problem.kind {
            ReduceBenchKind::Single => (1, 1),
            ReduceBenchKind::TwoLaunch => (2, 2),
            ReduceBenchKind::Fused => (1, 2),
        };

        Some(CategoryWork {
            compute_ops: 0,
            dtype,
            bytes_read: read_passes * input_elems * elem_size,
            bytes_written: write_halves * output_elems * elem_size,
        })
    }
}
