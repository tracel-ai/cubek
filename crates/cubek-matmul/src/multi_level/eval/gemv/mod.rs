mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::GemvCorrectness;
pub use problem::{GemvProblem, ProblemKind, problems};
pub use strategy::strategies;

use crate::definition::{MatmulCost, MatmulGlobalElems};
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

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

        // A gemv is a matmul whose m is one, so the kernel's own cost model
        // covers it and picks the compute probe with it.
        let cost = MatmulCost {
            batches: problem.batches,
            m: 1,
            n: problem.out_dim,
            k: problem.k_dim,
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
