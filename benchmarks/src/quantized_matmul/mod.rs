mod benchmark;
mod problem;
mod strategy;

pub use problem::{Layout, Mode, QuantSide, QuantizedMatmulProblem, problems};
pub use strategy::strategies;

use cubek::matmul::launch::Strategy;

use crate::registry::{CatalogEntry, RunSamples};

pub struct Category;

impl crate::registry::Category for Category {
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
        benchmark::bench(strategy, problem, num_samples)
    }
}
