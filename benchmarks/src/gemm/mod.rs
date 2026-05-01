mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::run;
pub use problem::problems;
pub use strategy::strategies;

use crate::registry::{
    BenchmarkCategory, CorrectnessOutcome, ItemDescriptor, ReferenceMode, RunSamples,
};

pub struct Category;

impl BenchmarkCategory for Category {
    fn id(&self) -> &'static str {
        "gemm"
    }
    fn label(&self) -> &'static str {
        "GEMM"
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

    fn correctness_cpu(
        &self,
        strategy_id: &str,
        problem_id: &str,
        epsilon: f32,
    ) -> Option<Result<CorrectnessOutcome, String>> {
        Some(correctness::cpu(strategy_id, problem_id, epsilon))
    }

    fn correctness_against_reference(
        &self,
        strategy_id: &str,
        problem_id: &str,
        mode: ReferenceMode,
        epsilon: f32,
    ) -> Option<Result<CorrectnessOutcome, String>> {
        Some(correctness::against_reference(strategy_id, problem_id, mode, epsilon))
    }
}
