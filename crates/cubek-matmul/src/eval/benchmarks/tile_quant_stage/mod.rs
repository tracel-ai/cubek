//! The tile DSL's packed-quant shared-memory stage on the register leaf, swept over stage depth.
//!
//! A quantized operand under the register leaf stages its *packed* storage words and unpacks at the
//! read. At an equal depth that ties with the dequantized stage the cmma leaf needs (both stream the
//! same bytes from global memory, and the unpack is off the reuse path). It pays by being ~4x
//! smaller, which buys depth — so depth is the axis worth sweeping. Drives the DSL directly rather
//! than a routine: quantization has no tile-DSL matmul routine yet.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{TileQuantStageProblem, problems};
pub use strategy::{StageDepth, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

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
}
