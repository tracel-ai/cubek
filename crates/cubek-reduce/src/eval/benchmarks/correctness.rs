use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::ReduceStrategy;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::cpu_reference::{
    cpu_reference_result, strategy_result, strategy_result_with_indices,
};

pub struct ReduceCorrectness;

impl cubek_test_utils::Correctness for ReduceCorrectness {
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
        // Validate the path that is actually benchmarked. The two-launch kind
        // runs the same `reduce` as `Single` for its values half, so only the
        // fused kind needs the dedicated entrypoint.
        match problem.kind {
            ReduceBenchKind::Single | ReduceBenchKind::TwoLaunch => strategy_result(
                client,
                problem.shape.clone(),
                problem.axis,
                strategy.clone(),
                problem.config,
                seeds[0],
            ),
            ReduceBenchKind::Fused => strategy_result_with_indices(
                client,
                problem.shape.clone(),
                problem.axis,
                strategy.clone(),
                problem.config,
                seeds[0],
            ),
        }
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
