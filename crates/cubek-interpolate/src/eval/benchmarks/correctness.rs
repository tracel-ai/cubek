use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::{
    definition::InterpolateProblem,
    eval::cpu_reference::{cpu_reference_result, strategy_result},
};

use super::InterpolateBenchmarkStrategy;

pub struct InterpolateCorrectness;

impl cubek_test_utils::Correctness for InterpolateCorrectness {
    type Problem = InterpolateProblem;
    type Strategy = InterpolateBenchmarkStrategy;

    fn kernel_result(
        &self,
        strategy: &InterpolateBenchmarkStrategy,
        problem: &InterpolateProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        match strategy {
            InterpolateBenchmarkStrategy::Standard(strategy) => {
                strategy_result(client, problem.clone(), *strategy, seeds[0])
            }
            InterpolateBenchmarkStrategy::Tile(_) => Err(
                "tile correctness is covered by focused tile tests; Lanczos3 border normalization is pending"
                    .to_string(),
            ),
        }
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
