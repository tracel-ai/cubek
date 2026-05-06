//! Seeded HostData primitives for the GEMM category.
//!
//! Both methods build the same input bits from `(strategy, problem, seeds[0..2])`
//! so the two `HostData`s they return are directly comparable.

use cubecl::{Runtime, TestRuntime, ir::AddressType, zspace::Shape};
use cubek::{
    matmul::{
        cpu_reference::{cpu_reference_result, strategy_result},
        definition::{MatmulElems, MatmulProblem},
        launch::Strategy,
    },
    std::MatrixLayout as KernelMatrixLayout,
};
use cubek_test_utils::{HostData, Progress};

use crate::gemm::problem::{GemmProblem, Precision};

pub struct GemmCorrectness;

impl crate::registry::Correctness for GemmCorrectness {
    type Problem = GemmProblem;
    type Strategy = Strategy;

    fn kernel_result(
        &self,
        strategy: &Strategy,
        problem: &GemmProblem,
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
        problem: &GemmProblem,
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

fn build_matmul_problem(p: &GemmProblem) -> MatmulProblem {
    let global_dtypes = match p.precision {
        Precision::F32 => MatmulElems::from_single_dtype(
            <f32 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked(),
        )
        .as_global_elems(),
        Precision::F16 => MatmulElems::from_single_dtype(
            <half::f16 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked(),
        )
        .as_global_elems(),
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
