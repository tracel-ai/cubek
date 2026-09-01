//! Benchmark catalogue for `cubek-convolution`.
//!
//! Gated behind the `benchmarks` cargo feature. The top-level `benchmarks`
//! crate re-exports [`Category`] from here and aggregates it with the other
//! kernels' catalogues.

mod benchmark;
mod correctness;
pub mod depthwise;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::Conv2dCorrectness;
pub use problem::{Conv2dProblem, problems};
pub use strategy::strategies;

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

use cubek_matmul::definition::MatmulGlobalElems;

use crate::definition::Conv2dCost;

use crate::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = Conv2dProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "conv2d"
    }

    fn label(&self) -> &'static str {
        "Conv2d"
    }

    fn problems(&self) -> Vec<CatalogEntry<Conv2dProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &Conv2dProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = Conv2dProblem, Strategy = Strategy>>
    {
        Some(&Conv2dCorrectness)
    }

    /// Standard conv flop count: one multiply-add per (output element, kernel tap,
    /// input channel). Runs in `half::f16`, the precision `bench` fixes.
    fn work(&self, problem: &Conv2dProblem) -> Option<CategoryWork> {
        let dtype = half::f16::elem_type_native();

        let [n, c_in, h_in, w_in] = problem.input_shape;
        let [c_out, _, k_h, k_w] = problem.weight_shape;
        let [s_h, s_w] = problem.args.stride;
        let [p_h, p_w] = problem.args.padding;
        let [d_h, d_w] = problem.args.dilation;

        let cost = Conv2dCost {
            batch: n,
            channels_in: c_in,
            spatial_in: [h_in, w_in],
            channels_out: c_out,
            kernel: [k_h, k_w],
            spatial_out: [
                (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1,
                (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1,
            ],
            bias_elems: problem.bias_shape,
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
