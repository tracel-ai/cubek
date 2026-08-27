mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{Layout, Mode, QuantSide, QuantizedMatmulProblem, problems};
pub use strategy::strategies;

use crate::definition::{MatmulCost, MatmulGlobalElems};
use cubecl::prelude::*;
use cubek_quant::scheme::QuantScheme;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

use crate::strategy::Strategy;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = QuantizedMatmulProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "quantized_matmul"
    }

    fn label(&self) -> &'static str {
        "Quantized Matmul"
    }

    fn problems(&self) -> Vec<CatalogEntry<QuantizedMatmulProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &QuantizedMatmulProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    /// The compute count is the same dot-product term count either way (the launch
    /// always runs `matmul_elems::<f32>()`), so only the read side changes: a
    /// quantized operand moves its packed data plus its scales instead of a plain
    /// float buffer.
    fn work(&self, problem: &QuantizedMatmulProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let lhs_shape = [problem.b, problem.m, problem.k];
        let rhs_shape = [problem.b, problem.k, problem.n];
        let (lhs_scheme, rhs_scheme) = match &problem.mode {
            Mode::Float => (None, None),
            Mode::Quant { scheme, side } => match side {
                QuantSide::LhsOnly => (Some(scheme), None),
                QuantSide::RhsOnly => (None, Some(scheme)),
                QuantSide::Both => (Some(scheme), Some(scheme)),
            },
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

        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(&client()),
            }),
            bytes_read: operand_bytes(&lhs_shape, lhs_scheme)
                + operand_bytes(&rhs_shape, rhs_scheme),
            bytes_written: problem.b * problem.m * problem.n * dtype.size(),
        })
    }
}

/// Bytes moved for one operand: a plain float buffer, or (quantized) its packed
/// `u32` data plus its `f32` scales.
fn operand_bytes(shape: &[usize], scheme: Option<&QuantScheme>) -> usize {
    let f32_size = f32::elem_type_native().size();
    match scheme {
        None => shape.iter().product::<usize>() * f32_size,
        Some(scheme) => {
            let numel: usize = shape.iter().product();
            let data_bytes = numel.div_ceil(scheme.num_quants()) * u32::elem_type_native().size();
            let scale_numel: usize = benchmark::scales_shape(scheme, shape).iter().product();
            data_bytes + scale_numel * f32_size
        }
    }
}
