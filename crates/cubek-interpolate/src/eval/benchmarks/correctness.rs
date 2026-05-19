use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::{
    definition::InterpolateProblem,
    eval::{
        benchmarks::strategy::InterpolateStrategy,
        cpu_reference::{cpu_reference_result, strategy_result},
    },
};

pub struct InterpolateCorrectness;

impl cubek_test_utils::Correctness for InterpolateCorrectness {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateStrategy;

    fn kernel_result(
        &self,
        _strategy: &InterpolateStrategy,
        problem: &InterpolateProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        strategy_result(client, problem.clone(), seeds[0])
    }

    fn reference_result(
        &self,
        problem: &InterpolateProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        cpu_reference_result(client, problem.clone(), seeds[0], progress)
    }
}
