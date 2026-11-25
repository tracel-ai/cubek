use cubecl::benchmark::TimingMethod;
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use super::{ReduceBenchKind, ReduceCorrectness, ReduceProblem, bench, problems, strategies};
use crate::eval::cpu_reference::{accumulation_dtype, output_dtype_for};
use crate::kernels::ReduceOpStrategy;

/// The reduce family's entry in the benchmark catalogue: its problems, its strategies, and what
/// one run reads, writes and computes.
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

    /// Every value read once and accumulated once and every cell written once a launch, and the
    /// index beside it where the problem writes both halves: a two-launch reduction reads and
    /// accumulates twice, which is the harness's doing.
    fn work(&self, problem: &ReduceProblem) -> Option<CategoryWork> {
        let value_dtype = problem.precision.to_dtype();
        let values: usize = problem.shape.iter().product();
        let cells = values / problem.shape[problem.axis] * problem.operation.written_per_row();
        let launches = match problem.kind {
            ReduceBenchKind::Single | ReduceBenchKind::Fused => 1,
            ReduceBenchKind::TwoLaunch => 2,
        };
        let cell_bytes = match problem.kind {
            ReduceBenchKind::Single => output_dtype_for(problem.operation, value_dtype).size(),
            ReduceBenchKind::TwoLaunch | ReduceBenchKind::Fused => {
                value_dtype.size() + u32::elem_type_native().size()
            }
        };

        Some(CategoryWork {
            compute: Some(ComputeWork::direct(launches * values, accumulation_dtype())),
            bytes_read: launches * values * value_dtype.size(),
            bytes_written: cells * cell_bytes,
        })
    }
}
