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
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::{InterpolateStrategy, definition::InterpolateProblem};

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

    /// No honest compute count: taps per output pixel range from 1 (nearest) to 36
    /// (lanczos3), and the backward pass' scatter has no fixed per-pixel cost either.
    /// Reads and writes are the tensor element counts.
    fn work(&self, problem: &InterpolateProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();

        let (read_elems, written_elems) = match problem {
            InterpolateProblem::Forward(prob) => (
                prob.input_shape().num_elements(),
                prob.output_shape().num_elements(),
            ),
            InterpolateProblem::Backward(prob) => {
                let [n, _, _, c] = prob.out_grad_shape;
                let input_grad_elems = n * prob.input_size[0] * prob.input_size[1] * c;
                (prob.out_grad_shape.iter().product(), input_grad_elems)
            }
        };

        Some(CategoryWork {
            compute_ops: 0,
            dtype,
            bytes_read: read_elems * elem_size,
            bytes_written: written_elems * elem_size,
        })
    }
}
