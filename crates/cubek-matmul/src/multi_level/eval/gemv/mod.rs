mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::GemvCorrectness;
pub use problem::{GemvProblem, ProblemKind, problems};
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::strategy::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = GemvProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "gemv"
    }

    fn label(&self) -> &'static str {
        "GEMV"
    }

    fn problems(&self) -> Vec<CatalogEntry<GemvProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &GemvProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = GemvProblem, Strategy = Strategy>>
    {
        Some(&GemvCorrectness)
    }

    /// The vector operand's `1` axis and the matrix operand's `out_dim × k_dim` axes
    /// swap roles between `VecMat` and `MatVec`, but the element counts read and
    /// written are the same either way.
    fn work(&self, problem: &GemvProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();
        let (b, out_dim, k) = (problem.batches, problem.out_dim, problem.k_dim);

        Some(CategoryWork {
            compute_ops: 2 * b * out_dim * k,
            dtype,
            bytes_read: (b * k + b * k * out_dim) * elem_size,
            bytes_written: b * out_dim * elem_size,
        })
    }
}
