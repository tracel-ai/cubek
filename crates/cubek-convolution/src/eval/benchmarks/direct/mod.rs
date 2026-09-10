mod benchmark;
mod problem;

pub use benchmark::bench;
pub use problem::{DirectProblem, DirectStrategy, problems, strategies};

use cubecl::prelude::*;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use crate::definition::Conv2dCost;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = DirectProblem;
    type Strategy = DirectStrategy;

    fn id(&self) -> &'static str {
        "direct"
    }

    fn label(&self) -> &'static str {
        "Direct conv2d"
    }

    fn problems(&self) -> Vec<CatalogEntry<DirectProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<DirectStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &DirectStrategy,
        problem: &DirectProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    fn work(&self, problem: &DirectProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let out = problem.out_size();

        let cost = Conv2dCost {
            batch: problem.batch,
            channels_in: problem.channels_in,
            spatial_in: [problem.size; 2],
            channels_out: problem.channels_out,
            kernel: [problem.kernel; 2],
            spatial_out: [out; 2],
            bias_elems: 0,
            elems: MatmulGlobalElems {
                lhs: dtype,
                rhs: dtype,
                out: dtype,
            },
        };

        let (bytes_read, bytes_written) = cost.traffic();

        Some(CategoryWork {
            // This routine issues no MMA: judge it against the scalar peak, not `compute_key`.
            compute: Some(ComputeWork::direct(cost.compute_ops(), dtype)),
            bytes_read,
            bytes_written,
        })
    }
}
