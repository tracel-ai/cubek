//! Seeded HostData primitives for the gemv category.
//!
//! Gemv is a special case of matmul (one inner dim is 1), so we build a
//! `MatmulProblem` and route through `cubek_matmul::cpu_reference`. Both
//! `kernel_result` and `reference_result` build the same input bits from
//! `(strategy_id, problem_id, seed_lhs, seed_rhs)`, so they're directly
//! comparable.

use cubecl::{
    Runtime, TestRuntime, ir::AddressType, ir::MatrixLayout as IrMatrixLayout, prelude::*,
    zspace::Shape,
};
use cubek::{
    matmul::{
        cpu_reference::{cpu_reference_result, strategy_result},
        definition::{MatmulElems, MatmulProblem},
    },
    std::MatrixLayout,
};
use cubek_test_utils::HostData;

use crate::gemv::{
    problem::{GemvProblem, ProblemKind, problem_for},
    strategy::strategy_for,
};

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let matmul_problem = build_matmul_problem(&problem);
    strategy_result(client, matmul_problem, strategy, seed_lhs, seed_rhs)
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
    let matmul_problem = build_matmul_problem(&problem);
    cpu_reference_result(client, matmul_problem, seed_lhs, seed_rhs)
}

fn build_matmul_problem(p: &GemvProblem) -> MatmulProblem {
    let (m, n, k) = match p.kind {
        ProblemKind::VecMat => (1, p.out_dim, p.k_dim),
        ProblemKind::MatVec => (p.out_dim, 1, p.k_dim),
    };
    let global_dtypes =
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems();
    MatmulProblem::from_parameters(
        m,
        n,
        k,
        Shape::from(vec![p.batches]),
        Shape::from(vec![p.batches]),
        ir_layout_to_matrix_layout(p.lhs_layout),
        ir_layout_to_matrix_layout(p.rhs_layout),
        MatrixLayout::RowMajor,
        None,
        None,
        global_dtypes,
        AddressType::U32,
    )
}

fn ir_layout_to_matrix_layout(layout: IrMatrixLayout) -> MatrixLayout {
    match layout {
        IrMatrixLayout::RowMajor => MatrixLayout::RowMajor,
        IrMatrixLayout::ColMajor => MatrixLayout::ColMajor,
        IrMatrixLayout::Undefined => panic!("undefined matrix layout"),
    }
}
