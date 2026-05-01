mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::run;
pub use problem::problems;
pub use strategy::strategies;

use cubek_test_utils::HostData;

use crate::registry::{BenchmarkCategory, ItemDescriptor, RunSamples};

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

    fn produce_kernel(
        &self,
        strategy_id: &str,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        Some(correctness::produce_kernel(strategy_id, problem_id, seed_lhs, seed_rhs))
    }

    fn produce_reference(
        &self,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        Some(correctness::produce_reference(problem_id, seed_lhs, seed_rhs))
    }
}
