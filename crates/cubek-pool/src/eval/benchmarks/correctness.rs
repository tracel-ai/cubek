use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::definition::PoolProblem;
use crate::eval::benchmarks::strategy::PoolStrategy;
use crate::eval::cpu_reference::{cpu_reference_result, strategy_result};

pub struct PoolCorrectness;

impl cubek_test_utils::Correctness for PoolCorrectness {
    type Problem = PoolProblem;
    type Strategy = PoolStrategy;

    fn kernel_result(
        &self,
        _strategy: &PoolStrategy,
        problem: &PoolProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        strategy_result(client, problem.clone(), seeds[0])
    }

    fn reference_result(
        &self,
        problem: &PoolProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        cpu_reference_result(client, problem.clone(), seeds[0], progress)
    }
}
