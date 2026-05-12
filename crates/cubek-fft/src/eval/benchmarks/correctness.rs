use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::eval::benchmarks::problem::FftProblem;
use crate::eval::benchmarks::strategy::FftStrategy;
use crate::eval::cpu_reference::{cpu_reference_result, kernel_result as fft_kernel_result};

pub struct FftCorrectness;

impl cubek_test_utils::Correctness for FftCorrectness {
    type Problem = FftProblem;
    type Strategy = FftStrategy;

    fn kernel_result(
        &self,
        _strategy: &FftStrategy,
        problem: &FftProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        fft_kernel_result(
            client,
            problem.shape.clone(),
            dim,
            problem.mode,
            seeds[0],
            seeds[1],
        )
    }

    fn reference_result(
        &self,
        problem: &FftProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        cpu_reference_result(
            client,
            problem.shape.clone(),
            dim,
            problem.mode,
            seeds[0],
            seeds[1],
            progress,
        )
    }
}
