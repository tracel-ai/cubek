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
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use crate::ReduceStrategy;
use crate::eval::cpu_reference::output_dtype_for;
use crate::launch::ReduceDtypes;
use crate::routines::ReduceCost;

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

    fn timing_method(&self) -> TimingMethod {
        TimingMethod::Device
    }

    fn correctness(
        &self,
    ) -> Option<
        &dyn cubek_test_utils::Correctness<Problem = ReduceProblem, Strategy = ReduceStrategy>,
    > {
        Some(&ReduceCorrectness)
    }

    /// Scaled by the passes the harness makes over the cost model: a two-launch
    /// reduction reads and writes twice, which is the harness's doing.
    fn work(&self, problem: &ReduceProblem) -> Option<CategoryWork> {
        let value_dtype = problem.precision.dtype();
        let input_elems: usize = problem.shape.iter().product();
        let reduce_len = problem.shape[problem.axis];

        let cost = ReduceCost {
            reduce_len,
            reduce_count: input_elems / reduce_len,
            instruction: problem.config,
            dtypes: ReduceDtypes {
                input: value_dtype,
                output: output_dtype_for(&problem.config, value_dtype),
                accumulation: f32::elem_type_native(),
            },
        };

        let (read_passes, write_passes) = match problem.kind {
            ReduceBenchKind::Single => (1, 1),
            ReduceBenchKind::TwoLaunch => (2, 2),
            ReduceBenchKind::Fused => (1, 2),
        };

        let (bytes_read, bytes_written) = cost.traffic();

        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(),
            }),
            bytes_read: read_passes * bytes_read,
            bytes_written: write_passes * bytes_written,
        })
    }
}
