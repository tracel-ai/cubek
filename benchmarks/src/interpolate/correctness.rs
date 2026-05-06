//! Seeded HostData primitives for the interpolate category.
//!
//! Both methods build the same input bits from `(problem, seeds[0])` —
//! interpolate is a unary op so `seeds[1..]` is ignored — and the two
//! `HostData`s they return are directly comparable.

use cubecl::{Runtime, TestRuntime};
use cubek::interpolate::{
    cpu_reference::{cpu_reference_result, strategy_result},
    definition::InterpolateProblem,
};
use cubek_test_utils::{HostData, Progress};

use crate::interpolate::strategy::InterpolateStrategy;

pub struct InterpolateCorrectness;

impl crate::registry::Correctness for InterpolateCorrectness {
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
