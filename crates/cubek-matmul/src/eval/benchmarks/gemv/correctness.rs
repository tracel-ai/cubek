use cubecl::{
    Runtime, TestRuntime, ir::AddressType, ir::MatrixLayout as IrMatrixLayout, prelude::*,
    zspace::Shape,
};
use cubek_std::MatrixLayout;
use cubek_test_utils::{HostData, Progress};

use crate::definition::{MatmulElems, MatmulProblem};
use crate::eval::benchmarks::gemv::problem::{GemvProblem, ProblemKind};
use crate::eval::cpu_reference::{cpu_reference_result, strategy_result};
use crate::strategy::Strategy;

pub struct GemvCorrectness;

impl cubek_test_utils::Correctness for GemvCorrectness {
    type Problem = GemvProblem;
    type Strategy = Strategy;

    fn kernel_result(
        &self,
        strategy: &Strategy,
        problem: &GemvProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        strategy_result(
            client,
            build_matmul_problem(problem),
            strategy.clone(),
            seeds[0],
            seeds[1],
        )
    }

    fn reference_result(
        &self,
        problem: &GemvProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        cpu_reference_result(
            client,
            build_matmul_problem(problem),
            seeds[0],
            seeds[1],
            progress,
        )
    }
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
