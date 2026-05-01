mod benchmark;
#[cfg(feature = "cpu-reference")]
mod correctness;
mod problem;
mod strategy;

pub use benchmark::run;
pub use problem::problems;
pub use strategy::strategies;

#[cfg(feature = "cpu-reference")]
use cubek_test_utils::HostData;

use crate::registry::{BenchmarkCategory, ItemDescriptor, RunSamples};

pub struct Category;

impl BenchmarkCategory for Category {
    fn id(&self) -> &'static str {
        "conv2d"
    }
    fn label(&self) -> &'static str {
        "Conv2d"
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

    #[cfg(feature = "cpu-reference")]
    fn kernel_result(
        &self,
        strategy_id: &str,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        Some(correctness::kernel_result(
            strategy_id,
            problem_id,
            seed_lhs,
            seed_rhs,
        ))
    }

    #[cfg(feature = "cpu-reference")]
    fn reference_result(
        &self,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        Some(correctness::reference_result(
            problem_id, seed_lhs, seed_rhs,
        ))
    }
}
