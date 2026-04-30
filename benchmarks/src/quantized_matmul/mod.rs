mod benchmark;
mod problem;
mod strategy;

pub use benchmark::run;
pub use problem::problems;
pub use strategy::strategies;

use crate::registry::{BenchmarkCategory, ItemDescriptor, RunSamples};

pub struct Category;

impl BenchmarkCategory for Category {
    fn id(&self) -> &'static str {
        "quantized_matmul"
    }
    fn label(&self) -> &'static str {
        "Quantized Matmul"
    }
    fn strategies(&self) -> Vec<ItemDescriptor> {
        strategies()
    }
    fn problems(&self) -> Vec<ItemDescriptor> {
        problems()
    }
    fn run(
        &self,
        strategy_id: &str,
        problem_id: &str,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        run(strategy_id, problem_id, num_samples)
    }
}
