//! Benchmark catalogue for `cubek-interpolate`.
mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::InterpolateCorrectness;
pub use problem::{problems, problems_scaled};
pub use strategy::{BenchTarget, BenchTier, every_strategy, strategies, strategies_at};

use cubecl::benchmark::TimingMethod;
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use crate::{
    InterpolateStrategy,
    definition::{InterpolateCost, InterpolateProblem},
};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateStrategy;

    fn id(&self) -> &'static str {
        "interpolate"
    }

    fn label(&self) -> &'static str {
        "Interpolate"
    }

    fn problems(&self) -> Vec<CatalogEntry<InterpolateProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<InterpolateStrategy>> {
        strategies(BenchTarget::Gpu)
    }

    fn bench(
        &self,
        strategy: &InterpolateStrategy,
        problem: &InterpolateProblem,
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
        &dyn cubek_test_utils::Correctness<
            Problem = InterpolateProblem,
            Strategy = InterpolateStrategy,
        >,
    > {
        Some(&InterpolateCorrectness)
    }
}

/// Every spatial extent divided by four, so one CPU sweep of the full strategy catalogue finishes
/// in minutes rather than hours. Sixteen times fewer output elements per problem, with the batch,
/// channel and scale-direction regimes left intact.
const CPU_DIVISOR: usize = 4;

/// The same sweep at CPU scale.
///
/// It is a separate category rather than a switch on the device because the two are not
/// comparable: a median from this set says which strategy wins, not what the operation costs.
pub struct CpuCategory;

impl cubek_test_utils::Category for CpuCategory {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateStrategy;

    fn id(&self) -> &'static str {
        "interpolate_cpu"
    }

    fn label(&self) -> &'static str {
        "Interpolate (CPU scale)"
    }

    fn problems(&self) -> Vec<CatalogEntry<InterpolateProblem>> {
        problems_scaled(CPU_DIVISOR)
    }

    fn strategies(&self) -> Vec<CatalogEntry<InterpolateStrategy>> {
        strategies(BenchTarget::Cpu)
    }

    fn bench(
        &self,
        strategy: &InterpolateStrategy,
        problem: &InterpolateProblem,
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
        &dyn cubek_test_utils::Correctness<
            Problem = InterpolateProblem,
            Strategy = InterpolateStrategy,
        >,
    > {
        Some(&InterpolateCorrectness)
    }

    fn work(&self, problem: &InterpolateProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let cost = InterpolateCost::new(problem.clone(), dtype);
        let (bytes_read, bytes_written) = cost.traffic();

        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(),
            }),
            bytes_read,
            bytes_written,
        })
    }
}
