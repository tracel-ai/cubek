//! Seeded "produce a HostData" primitives for the GEMM category.
//!
//! Both `produce_kernel` and `produce_reference` build the same input bits
//! from `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the two
//! `HostData`s they return are directly comparable. Persistence and
//! file-vs-file comparison live entirely in the tuner — this module knows
//! nothing about disk.

use cubecl::{
    Runtime, TestRuntime,
    ir::AddressType,
    zspace::{Shape, shape},
};
use cubek::{
    matmul::{
        cpu_reference::{produce_cpu_reference_result, produce_strategy_result},
        definition::{MatmulElems, MatmulPrecision, MatmulProblem},
    },
    std::MatrixLayout as KernelMatrixLayout,
};
use cubek_test_utils::HostData;

use crate::gemm::{
    problem::{GemmProblem, Precision, problem_for},
    strategy::strategy_for,
};

pub fn produce_kernel(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem = problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let matmul_problem = build_matmul_problem(&problem);
    produce_strategy_result(client, matmul_problem, strategy, seed_lhs, seed_rhs)
}

pub fn produce_reference(
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem = problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let matmul_problem = build_matmul_problem(&problem);
    produce_cpu_reference_result(client, matmul_problem, seed_lhs, seed_rhs)
}

fn build_matmul_problem(p: &GemmProblem) -> MatmulProblem {
    let global_dtypes = match p.precision {
        Precision::F32 => MatmulElems::from_single_dtype(<f32 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked()).as_global_elems(),
        Precision::F16 => MatmulElems::from_single_dtype(<half::f16 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked()).as_global_elems(),
    };
    MatmulProblem::from_parameters(
        p.m,
        p.n,
        p.k,
        Shape::from(vec![p.b]),
        Shape::from(vec![p.b]),
        p.lhs_layout,
        p.rhs_layout,
        KernelMatrixLayout::RowMajor,
        None,
        None,
        global_dtypes,
        AddressType::U32,
    )
}

#[allow(dead_code)]
fn _unused() {
    let _ = shape![1];
    let _: Precision = Precision::F32;
}

#[allow(dead_code)]
fn _phantom<MP: MatmulPrecision>() {}
