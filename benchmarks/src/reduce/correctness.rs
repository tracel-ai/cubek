//! Seeded HostData primitives for the reduce category.
//!
//! Both `kernel_result` and `reference_result` build the same input bits from
//! `(strategy_id, problem_id, seed_lhs)` (the `seed_rhs` parameter is unused
//! since reduce is a unary operation) so the two `HostData`s they return are
//! directly comparable.

use cubecl::{Runtime, TestRuntime};
use cubek::reduce::cpu_reference::{cpu_reference_result, strategy_result};
use cubek_test_utils::HostData;

use crate::reduce::{problem::problem_for, strategy::strategy_for};

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    _seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    strategy_result(
        client,
        problem.shape,
        problem.axis,
        strategy,
        problem.config,
        seed_lhs,
    )
}

pub fn reference_result(
    problem_id: &str,
    seed_lhs: u64,
    _seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    cpu_reference_result(
        client,
        problem.shape,
        problem.axis,
        problem.config,
        seed_lhs,
    )
}
