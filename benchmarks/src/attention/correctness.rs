//! Seeded HostData primitives for the attention category.
//!
//! Both `kernel_result` and `reference_result` build the same input bits
//! from `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the two
//! `HostData`s they return are directly comparable.

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek::attention::{
    cpu_reference::{cpu_reference_result, strategy_result},
    definition::AttentionGlobalTypes,
};
use cubek_test_utils::HostData;

use crate::attention::{problem::problem_for, strategy::strategy_for};

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let problem = build_problem(problem_id, &client)?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    strategy_result(client, problem, strategy, seed_lhs, seed_rhs)
}

pub fn reference_result(
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let problem = build_problem(problem_id, &client)?;
    cpu_reference_result(client, problem, seed_lhs, seed_rhs)
}

fn build_problem(
    problem_id: &str,
    client: &cubecl::client::ComputeClient<TestRuntime>,
) -> Result<cubek::attention::definition::AttentionProblem, String> {
    let global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(client),
    );
    problem_for(problem_id, global_dtypes).ok_or_else(|| format!("unknown problem: {problem_id}"))
}
