mod benchmark;
mod problem;

pub use benchmark::bench;
pub use problem::{DirectProblem, DirectStrategy, problems, strategies};

use cubecl::prelude::*;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples};

use crate::definition::{ConvCost, ConvCostDim};

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

        let cost = ConvCost {
            batch: problem.batch,
            channels_in: problem.channels_in,
            channels_out: problem.channels_out,
            spatial: vec![
                ConvCostDim {
                    input: problem.size,
                    kernel: problem.kernel,
                    out,
                };
                2
            ],
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
