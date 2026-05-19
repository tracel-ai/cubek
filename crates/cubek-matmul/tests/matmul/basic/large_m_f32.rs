//! Regression test for a `Strategy::Auto` correctness failure on the wgpu
//! backend at large M with f32 inputs (reported against Burn 0.21 on Apple
//! Metal). For M ≤ 256 the auto-selected kernel matches the CPU reference at
//! noise floor; once M crosses ~512 the output collapses to (near-)zero RMS,
//! while the same shape splits into 64-row chunks recover correct output.
//!
//! Reproduced on Apple M1 (wgpu/Metal); the corresponding NVIDIA + wgpu run
//! is reportedly fine, so this is expected to fail on macOS only.

use cubecl::TestRuntime;
use cubecl::ir::AddressType;
use cubecl::prelude::*;
use cubecl::zspace::shape;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::components::tile::{TileMatmul, TileMatmulKind};
use cubek_matmul::definition::{
    MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError, TilingBlueprint,
    TilingScheme,
};
use cubek_matmul::launch::{Strategy, launch_ref};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_std::{
    InputBinding, MatrixLayout, PartitionSize, StageSize, TileSize,
};
use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

use super::common::{client, f16_elems, f32_elems, rect};
use crate::matmul::layout_to_stride_spec;
use crate::matmul::{assert_result, test_matmul_strategy};

/// Smallest known-broken shape from the bug report: (M=535, K=1024, N=1024)
/// row-major f32, `Strategy::Auto`.
#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f32_535x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(535, 1024, 1024, f32_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f32_1024x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(1024, 1024, 1024, f32_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f32_2048x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(2048, 1024, 1024, f32_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f16_535x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(535, 1024, 1024, f16_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f16_1024x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(1024, 1024, 1024, f16_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f16_2048x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(2048, 1024, 1024, f16_elems()),
        Strategy::Auto,
    );
}

#[cfg(feature = "heavy")]
#[test]
fn simple_unit_large_m_f32_535x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(535, 1024, 1024, f32_elems()),
        Strategy::SimpleUnit(Default::default()),
    );
}

#[cfg(feature = "heavy")]
#[test]
fn simple_unit_large_m_f32_2048x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(2048, 1024, 1024, f32_elems()),
        Strategy::SimpleUnit(Default::default()),
    );
}

#[cfg(feature = "heavy")]
#[test]
fn naive_large_m_f32_535x1024x1024() {
    test_matmul_strategy(
        client(),
        rect(535, 1024, 1024, f32_elems()),
        Strategy::Naive,
    );
}

/// Synthetic over-budget test: force a giant tile/partition/stage blueprint
/// so the kernel's lhs+rhs shared-memory footprint exceeds any reasonable
/// per-cube budget, and assert that
/// [`MatmulAvailabilityError::SharedMemoryTooBig`] is returned by the setup
/// path (before the kernel is ever queued).
#[test]
fn smem_budget_rejected_for_oversized_blueprint() {
    let c = client();
    let problem = MatmulProblem::from_parameters(
        128,
        128,
        128,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        f32_elems(),
        AddressType::U32,
    );

    // Pick a tile/partition/stage that guarantees an over-budget kernel on any
    // current hardware: 64x64x16 tile × 4x4x1 partition × 4x4x1 stage with f32
    // ⇒ several megabytes of shared memory for LHS alone.
    let scheme = TilingScheme::builder()
        .with_tile_size(TileSize {
            m: 64,
            n: 64,
            k: 64,
        })
        .with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
        // 8 × 4 = 32 units, matching plane_dim on Apple M1 so other validators
        // accept the config; smem footprint still vastly exceeds any budget.
        .with_stage_size(StageSize { m: 8, n: 4, k: 1 })
        .build()
        .unwrap();
    let plane_dim = c.properties().hardware.plane_size_max;
    let blueprint = TilingBlueprint::builder(TileMatmulKind::Register, scheme, plane_dim, &problem)
        .partition_buffering(PartitionBuffering::Single)
        .build();

    let (lhs, _) = TestInput::builder(c.clone(), problem.lhs_shape.clone())
        .dtype(problem.global_dtypes.lhs)
        .stride(layout_to_stride_spec(problem.lhs_layout))
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();
    let (rhs, _) = TestInput::builder(c.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .stride(layout_to_stride_spec(problem.rhs_layout))
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();
    let out = TestInput::builder(c.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .stride(layout_to_stride_spec(MatrixLayout::RowMajor))
        .zeros()
        .generate_without_host_data();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());
    let strategy = Strategy::SimpleUnit(BlueprintStrategy::Forced(blueprint));
    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.binding();

    let result = launch_ref(&strategy, &c, lhs_handle, rhs_handle, out_handle, &mut dtypes);
    match result {
        Err(MatmulSetupError::Unavailable(MatmulAvailabilityError::SharedMemoryTooBig {
            requested,
            available,
        })) => {
            assert!(
                requested > available,
                "SharedMemoryTooBig must report requested > available (got requested={requested}, available={available})"
            );
        }
        Err(other) => panic!(
            "expected SharedMemoryTooBig, got a different setup error: {other:?}"
        ),
        Ok(()) => panic!(
            "expected SharedMemoryTooBig, but launch succeeded for an over-budget blueprint"
        ),
    }
}

/// Mirrors the Burn 0.21 repro's input magnitudes: arange scaled by 1e-4,
/// producing output RMS in the millions (vs. ~10 with Uniform(-1,1)), in case
/// the bug is magnitude-dependent. Uses `Strategy::Auto` end-to-end.
#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f32_535x1024x1024_arange() {
    run_arange(535, 1024, 1024);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_large_m_f32_2048x1024x1024_arange() {
    run_arange(2048, 1024, 1024);
}

#[cfg(feature = "heavy")]
fn run_arange(m: usize, k: usize, n: usize) {
    let c = client();
    let mut problem = rect(m, n, k, f32_elems());

    let (lhs, lhs_host) = TestInput::builder(c.clone(), problem.lhs_shape.clone())
        .dtype(problem.global_dtypes.lhs)
        .stride(layout_to_stride_spec(problem.lhs_layout))
        .arange_scaled(1e-4)
        .generate_with_f32_host_data();

    let (rhs, rhs_host) = TestInput::builder(c.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .stride(layout_to_stride_spec(problem.rhs_layout))
        .arange_scaled(1e-4)
        .generate_with_f32_host_data();

    let out = TestInput::builder(c.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .stride(layout_to_stride_spec(MatrixLayout::RowMajor))
        .zeros()
        .generate_without_host_data();

    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());
    let strategy = Strategy::Auto;

    let outcome = launch_and_capture_outcome(&c, |client| {
        launch_ref(&strategy, client, lhs_handle, rhs_handle, out_handle, &mut dtypes).into()
    });

    match outcome {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_host, &rhs_host, &problem, &c, out, dtypes).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}
