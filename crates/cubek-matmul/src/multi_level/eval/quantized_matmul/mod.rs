mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{Layout, Mode, QuantSide, QuantizedMatmulProblem, problems};
pub use strategy::strategies;

use cubek_test_utils::{CatalogEntry, RunSamples};

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
}
