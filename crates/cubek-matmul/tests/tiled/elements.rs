//! A product written in another element than its inputs': the sum is cast to the output's
//! element as it drains, so the tile-DSL routines write any element they are verified at.

use crate::harness::test_matmul_strategy;
use cubecl::ir::{AddressType, ElemType, FloatKind};
use cubecl::zspace::shape;
use cubek_matmul::{
    definition::{MatmulGlobalElems, MatmulProblem},
    kernels::{CubeStagedMatmulRoutineSettings, UnitRegistersMatmulRoutineSettings},
    strategy::Strategy,
    tiled::Strategy as Tiled,
};
use cubek_std::MatrixLayout;

/// `f16` inputs, `f32` output: what a framework asks for when it keeps a product's output wider
/// than its operands.
fn f16_into_f32(strategy: Strategy) {
    let (f16, f32) = (
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
    );
    let problem = MatmulProblem::from_parameters(
        64,
        48,
        96,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulGlobalElems {
            lhs: f16,
            rhs: f16,
            out: f32,
        },
        AddressType::U32,
    );
    test_matmul_strategy(cubecl::test_device().client(), problem, strategy);
}

#[test]
fn unit_registers_writes_a_wider_output() {
    f16_into_f32(Tiled::CpuGemm(UnitRegistersMatmulRoutineSettings::DEFAULT).into());
}

#[test]
fn cube_staged_writes_a_wider_output() {
    f16_into_f32(Tiled::Cmma(CubeStagedMatmulRoutineSettings::DEFAULT).into());
}
