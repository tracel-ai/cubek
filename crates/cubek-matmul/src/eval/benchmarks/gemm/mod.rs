mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::GemmCorrectness;
pub use problem::{GemmProblem, Precision, problems};
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::strategy::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = GemmProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "gemm"
    }

    fn label(&self) -> &'static str {
        "GEMM"
    }

    fn problems(&self) -> Vec<CatalogEntry<GemmProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &GemmProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = GemmProblem, Strategy = Strategy>>
    {
        Some(&GemmCorrectness)
    }

    fn work(&self, problem: &GemmProblem) -> Option<CategoryWork> {
        let dtype = match problem.precision {
            Precision::F32 => f32::elem_type_native(),
            Precision::F16 => half::f16::elem_type_native(),
        };
        let elem_size = dtype.size();
        let (b, m, n, k) = (problem.b, problem.m, problem.n, problem.k);

        Some(CategoryWork {
            compute_ops: 2 * b * m * n * k,
            dtype,
            bytes_read: (b * m * k + b * k * n) * elem_size,
            bytes_written: b * m * n * elem_size,
        })
    }
}
