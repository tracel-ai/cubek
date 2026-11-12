use super::{Conv2dCorrectness, Conv2dProblem, bench, problems, strategies};
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
    /// input channel), in the element the problem states.
    fn work(&self, problem: &Conv2dProblem) -> Option<CategoryWork> {
        let dtype = problem.element.elem_type();

        let [n, c_in, h_in, w_in] = problem.input_shape;
        let [c_out, _, k_h, k_w] = problem.weight_shape;

        let cost = Conv2dCost {
            batch: n,
            channels_in: c_in,
            spatial_in: [h_in, w_in],
            channels_out: c_out,
            kernel: [k_h, k_w],
            spatial_out: problem.out_extent(),
            bias_elems: 0,
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
