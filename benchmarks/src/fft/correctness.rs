//! Seeded HostData primitives for the FFT category.
//!
//! Both methods build the same input bits from `(problem, seeds[0..2])` and
//! the two `HostData`s they return are directly comparable. Forward (RFFT)
//! returns the (re, im) pair stacked along a fresh leading dim of size 2 —
//! see [`cubek::fft::cpu_reference`] for details.

use cubecl::{Runtime, TestRuntime};
use cubek::fft::cpu_reference::{cpu_reference_result, kernel_result as fft_kernel_result};
use cubek_test_utils::{HostData, Progress};

use crate::fft::{problem::FftProblem, strategy::FftStrategy};

pub struct FftCorrectness;

impl crate::registry::Correctness for FftCorrectness {
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
