//! Seeded HostData primitives for the FFT category.
//!
//! Both `kernel_result` and `reference_result` build the same input bits
//! from `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the two
//! `HostData`s they return are directly comparable.
//!
//! Forward (RFFT) returns the (re, im) pair stacked along a fresh leading dim
//! of size 2 — see [`cubek::fft::cpu_reference`] for details.

use cubecl::{Runtime, TestRuntime};
use cubek::fft::cpu_reference::{cpu_reference_result, kernel_result as fft_kernel_result};
use cubek_test_utils::HostData;

use crate::fft::{problem::problem_for, strategy::strategy_for};

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dim = problem.shape.len() - 1;
    fft_kernel_result(client, problem.shape, dim, problem.mode, seed_lhs, seed_rhs)
}

pub fn reference_result(
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dim = problem.shape.len() - 1;
    cpu_reference_result(client, problem.shape, dim, problem.mode, seed_lhs, seed_rhs)
}
