use cubecl::benchmark::TimingMethod;
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use super::benchmark::bench;
use super::correctness::ReduceCorrectness;
use super::problem::{ReduceBenchKind, ReduceProblem, problems};
use super::strategy::strategies;
use crate::eval::benchmarks::results::{accumulation_dtype, output_dtype_for};
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

    /// # Panics
    ///
    /// Where `problem`'s axis is not one of its shape's, or it compares a values launch with an
    /// indices launch for an operation that has no such pair (a sum, a mean, a product, a max of
    /// absolute values, any or all).
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

    /// # Panics
    ///
    /// Where `problem`'s axis is not one of its shape's, or holds no value.
    fn work(&self, problem: &ReduceProblem) -> Option<CategoryWork> {
        Some(work_of(problem))
    }
}

/// Every value read once and accumulated once and every cell written once a launch, and the index
/// beside it where the problem writes both halves: a two-launch reduction reads and accumulates
/// twice, which is the harness's doing.
fn work_of(problem: &ReduceProblem) -> CategoryWork {
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

    CategoryWork {
        compute: Some(ComputeWork::direct(launches * values, accumulation_dtype())),
        bytes_read: launches * values * value_dtype.size(),
        bytes_written: cells * cell_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReduceOperation;
    use crate::eval::benchmarks::ReduceBenchPrecision;
    use crate::kernels::ReduceKernel;
    use cubek_test_utils::Category as _;

    /// A sum of four rows of 300 f32 values, launched as `kind` says.
    fn problem(operation: ReduceOperation, kind: ReduceBenchKind) -> ReduceProblem {
        ReduceProblem {
            shape: vec![4, 300],
            axis: 1,
            operation,
            kind,
            precision: ReduceBenchPrecision::F32,
        }
    }

    #[test]
    fn its_id_is_the_one_the_history_holds() {
        assert_eq!(Category.id(), "reduce");
    }

    #[test]
    fn its_label_is_the_familys() {
        assert_eq!(Category.label(), "Reduce");
    }

    #[test]
    fn it_lists_problems() {
        assert!(!Category.problems().is_empty());
    }

    #[test]
    fn it_lists_every_strategy() {
        assert_eq!(Category.strategies().len(), ReduceOpStrategy::ALL.len());
    }

    #[test]
    fn it_times_on_the_device() {
        let device = cubek_test_utils::timing_method(TimingMethod::Device);
        assert_eq!(Category.timing_method(), device);
    }

    #[test]
    fn it_checks_its_problems() {
        assert!(Category.correctness().is_some());
    }

    #[test]
    fn a_launch_reads_every_value() {
        let work = work_of(&problem(ReduceOperation::Sum, ReduceBenchKind::Single));
        assert_eq!(work.bytes_read, 4 * 300 * size_of::<f32>());
    }

    #[test]
    fn two_launches_read_every_value_twice() {
        let work = work_of(&problem(ReduceOperation::Max, ReduceBenchKind::TwoLaunch));
        assert_eq!(work.bytes_read, 2 * 4 * 300 * size_of::<f32>());
    }

    #[test]
    fn a_fused_launch_writes_a_value_and_its_index_a_cell() {
        let work = work_of(&problem(ReduceOperation::Max, ReduceBenchKind::Fused));
        assert_eq!(
            work.bytes_written,
            4 * (size_of::<f32>() + size_of::<u32>())
        );
    }

    #[test]
    fn a_bench_times_each_sample() {
        const SAMPLE_COUNT: usize = 10;
        let unit = ReduceOpStrategy::Specific(ReduceKernel::Unit);
        let problem = problem(ReduceOperation::Sum, ReduceBenchKind::Single);
        let samples = Category
            .bench(&unit, &problem, SAMPLE_COUNT)
            .expect("the unit kernel serves every problem");
        assert_eq!(samples.durations.len(), SAMPLE_COUNT);
    }
}
