//! Seeded HostData primitives for the interpolate category.
//!
//! Both `kernel_result` and `reference_result` build the same input bits
//! from `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the two
//! `HostData`s they return are directly comparable.

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek::interpolate::cpu_reference::{cpu_reference_result, strategy_result};
use cubek::interpolate::definition::InterpolateProblem;
use cubek_test_utils::{HostData, Progress};

use crate::interpolate::problem::problem_for;

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let problem = build_problem(problem_id, &client)?;
    strategy_result(client, problem, strategy, seed_lhs, seed_rhs)
}

pub fn reference_result(
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let problem = build_problem(problem_id, &client)?;
    cpu_reference_result(client, problem, seed_lhs, seed_rhs, progress)
}

fn build_problem(
    problem_id: &str,
    client: &cubecl::client::ComputeClient<TestRuntime>,
) -> Result<InterpolateProblem, String> {
    problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))
}
