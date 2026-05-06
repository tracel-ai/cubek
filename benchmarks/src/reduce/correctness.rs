//! Seeded HostData primitives for the reduce category.
//!
//! Both methods build the same input bits from `(problem, seeds[0])` —
//! reduce is unary so `seeds[1..]` is ignored — and the two `HostData`s
//! they return are directly comparable.

use cubecl::{Runtime, TestRuntime};
use cubek::reduce::{
    cpu_reference::{cpu_reference_result, strategy_result},
    launch::ReduceStrategy,
};
use cubek_test_utils::{HostData, Progress};

use crate::reduce::problem::ReduceProblem;

pub struct ReduceCorrectness;

impl crate::registry::Correctness for ReduceCorrectness {
    type Problem = ReduceProblem;
    type Strategy = ReduceStrategy;

    fn kernel_result(
        &self,
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        strategy_result(
            client,
            problem.shape.clone(),
            problem.axis,
            strategy.clone(),
            problem.config,
            seeds[0],
        )
    }

    fn reference_result(
        &self,
        problem: &ReduceProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        cpu_reference_result(
            client,
            problem.shape.clone(),
            problem.axis,
            problem.config,
            seeds[0],
            progress,
        )
    }
}
