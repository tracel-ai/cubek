//! The tile DSL's packed-quant shared-memory stage on the register leaf, swept over stage depth.
//!
//! A quantized operand under the register leaf stages its *packed* storage words and unpacks at the
//! read. At an equal depth that ties with the dequantized stage the cmma leaf needs (both stream the
//! same bytes from global memory, and the unpack is off the reuse path). It pays by being ~4x
//! smaller, which buys depth, so depth is the axis worth sweeping. Drives the DSL directly rather
//! than a routine: quantization has no tile-DSL matmul routine yet.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{TileQuantStageProblem, problems};
pub use strategy::{StageDepth, strategies};

use crate::definition::{MatmulCost, MatmulGlobalElems};
use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, client};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = TileQuantStageProblem;
    type Strategy = StageDepth;

    fn id(&self) -> &'static str {
        "tile_quant_stage"
    }

    fn label(&self) -> &'static str {
        "Tile: quantized smem stage"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<TileQuantStageProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<StageDepth>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &StageDepth,
        problem: &TileQuantStageProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    fn work(&self, problem: &TileQuantStageProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let f32_size = dtype.size();
        let scheme = benchmark::quant_scheme(problem.bn);

        let a_bytes = problem.m * problem.k * f32_size;
        let b_data_bytes =
            (problem.k * problem.n).div_ceil(scheme.num_quants()) * u32::elem_type_native().size();
        let b_scale_bytes = problem.k * (problem.n / problem.bn) * f32_size;
        let c_bytes = problem.m * problem.n * f32_size;

        let cost = MatmulCost {
            batches: 1,
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
            bytes_read: a_bytes + b_data_bytes + b_scale_bytes,
            bytes_written: c_bytes,
        })
    }
}
