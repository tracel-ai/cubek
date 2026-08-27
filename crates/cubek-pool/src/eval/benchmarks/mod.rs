//! Benchmark catalogue for `cubek-pool`.
mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::PoolCorrectness;
pub use problem::problems;
pub use strategy::{PoolStrategy, strategies};

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::definition::{PoolBackward, PoolForward, PoolProblem};
use crate::eval::cpu_reference::geometry::PoolGeometry;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = PoolProblem;
    type Strategy = PoolStrategy;

    fn id(&self) -> &'static str {
        "pool"
    }

    fn label(&self) -> &'static str {
        "Pool"
    }

    fn problems(&self) -> Vec<CatalogEntry<PoolProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<PoolStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &PoolStrategy,
        problem: &PoolProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = PoolProblem, Strategy = PoolStrategy>>
    {
        Some(&PoolCorrectness)
    }

    /// No honest compute count: the window size varies per mode and problem, and
    /// max/avg/adaptive-avg cost different amounts per window. Reads and writes are
    /// the tensor element counts, indices included where the problem asks for them.
    fn work(&self, problem: &PoolProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();

        match problem {
            PoolProblem::Forward(PoolForward::D2(prob)) => {
                let input_elems = prob.input_shape.num_elements();
                let output_elems = prob.output_shape(&prob.input_shape).num_elements();
                let indices_elems = if prob.with_indices { output_elems } else { 0 };

                Some(CategoryWork {
                    compute: None,
                    bytes_read: input_elems * elem_size,
                    bytes_written: (output_elems + indices_elems) * elem_size,
                })
            }
            PoolProblem::Backward(PoolBackward::D2(prob)) => {
                let n = prob.out_grad_shape[0];
                let c = prob.out_grad_shape[3];
                let out_grad_elems = prob.out_grad_shape.num_elements();
                let input_grad_elems = n * prob.input_size[0] * prob.input_size[1] * c;
                let indices_elems = if prob.with_indices { out_grad_elems } else { 0 };

                Some(CategoryWork {
                    compute: None,
                    bytes_read: (out_grad_elems + indices_elems) * elem_size,
                    bytes_written: input_grad_elems * elem_size,
                })
            }
            _ => None,
        }
    }
}
