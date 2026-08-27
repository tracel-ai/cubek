mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::GemmCorrectness;
pub use problem::{GemmProblem, Precision, problems};
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

use crate::definition::{MatmulCost, MatmulGlobalElems};
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

    /// The kernel's own cost model, which burn already scores autotune bounds
    /// against, rather than a second count of the same matmul.
    ///
    /// It also picks the compute probe, which matters here: a strategy issuing
    /// MMA runs on hardware the scalar peak does not describe, and judged
    /// against that peak the tensor-core strategies read 145 to 236% of it.
    fn work(&self, problem: &GemmProblem) -> Option<CategoryWork> {
        let dtype = match problem.precision {
            Precision::F32 => f32::elem_type_native(),
            Precision::F16 => half::f16::elem_type_native(),
        };

        let cost = MatmulCost {
            batches: problem.b,
            m: problem.m,
            n: problem.n,
            k: problem.k,
            elems: MatmulGlobalElems {
                lhs: dtype,
                rhs: dtype,
                out: dtype,
            },
        };

        let (bytes_read, bytes_written) = cost.traffic();

        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(&client()),
            }),
            bytes_read,
            bytes_written,
        })
    }
}
