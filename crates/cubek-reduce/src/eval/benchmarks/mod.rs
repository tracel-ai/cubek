//! Benchmark catalogue for `cubek-reduce`.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::ReduceCorrectness;
pub use problem::{ReduceBenchKind, ReduceBenchPrecision, ReduceProblem, precisions, problems};
pub use strategy::strategies;

use cubecl::benchmark::TimingMethod;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use crate::eval::cpu_reference::{accumulation_dtype, output_dtype_for};
use crate::kernels::ReduceOpStrategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = ReduceProblem;
    type Strategy = ReduceOpStrategy;

    fn id(&self) -> &'static str {
        "reduce"
    }

    fn label(&self) -> &'static str {
        "Reduce"
    }

    fn problems(&self) -> Vec<CatalogEntry<ReduceProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ReduceOpStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ReduceOpStrategy,
        problem: &ReduceProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    fn timing_method(&self) -> TimingMethod {
        cubek_test_utils::timing_method(TimingMethod::Device)
    }

    fn correctness(
        &self,
    ) -> Option<
        &dyn cubek_test_utils::Correctness<Problem = ReduceProblem, Strategy = ReduceOpStrategy>,
    > {
        Some(&ReduceCorrectness)
    }

    /// Every value read once and accumulated once, every cell written once, scaled by the passes
    /// the harness makes: a two-launch reduction reads and writes twice, which is the harness's
    /// doing.
    fn work(&self, problem: &ReduceProblem) -> Option<CategoryWork> {
        let value_dtype = problem.precision.dtype();
        let values: usize = problem.shape.iter().product();
        let cells = values / problem.shape[problem.axis] * problem.operation.written_per_row();
        let out_dtype = output_dtype_for(problem.operation, value_dtype);

        let (read_passes, write_passes) = match problem.kind {
            ReduceBenchKind::Single => (1, 1),
            ReduceBenchKind::TwoLaunch => (2, 2),
            ReduceBenchKind::Fused => (1, 2),
        };

        Some(CategoryWork {
            compute: Some(ComputeWork::direct(values, accumulation_dtype())),
            bytes_read: read_passes * values * value_dtype.size(),
            bytes_written: write_passes * cells * out_dtype.size(),
        })
    }
}
