//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubek_matmul::{
    multi_level::Strategy as MultiLevel,
    routine::BlueprintStrategy,
    strategy::Strategy,
    tiled::{Strategy as Tiled, cmma::CmmaStrategy},
};

use super::common::{client, f16_elems, square};
use crate::matmul::test_matmul_strategy;

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn simple_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridCmma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridMma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicMma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedMma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleCmma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleMma(Default::default()).into(),
    );
}

// ---- the tile-DSL port of the simple cyclic cmma matmul --------------------------------

#[test]
fn cmma_square_f32() {
    test_matmul_strategy(
        client(),
        square(256, super::common::f32_elems()),
        Tiled::Cmma(Default::default()).into(),
    );
}

#[test]
fn cmma_square_f16() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Tiled::Cmma(Default::default()).into(),
    );
}

#[test]
fn cmma_rect_f32() {
    test_matmul_strategy(
        client(),
        super::common::rect(64, 128, 32, super::common::f32_elems()),
        Tiled::Cmma(Default::default()).into(),
    );
}

#[test]
fn cmma_batched_f32() {
    use cubecl::{ir::AddressType, zspace::shape};
    use cubek_matmul::definition::MatmulProblem;
    use cubek_std::MatrixLayout;

    let elems = super::common::f32_elems();
    let problem = MatmulProblem::from_parameters(
        64,
        64,
        64,
        shape![3],
        shape![3],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    );
    test_matmul_strategy(client(), problem, Tiled::Cmma(Default::default()).into());
}

/// A plane owning exactly one fragment along both axes: the Staged level's cuts equal the
/// leaf's, so the fragment grid is 1×1. Regression test for the degenerate partition being
/// misread as an instance level, which staged the per-plane operand fragments into
/// cube-shared smem (every plane contracted plane 0's windows).
#[test]
fn cmma_partition_1x1_f32() {
    use cubek_matmul::tiled::{
        cmma::{CmmaBlueprint, CmmaDelivery, Partition},
        cpu_gemm::{InstructionShape, PlaneGrid},
    };

    let blueprint = CmmaBlueprint {
        instruction: InstructionShape { m: 8, n: 8, k: 8 },
        partition: Partition { m: 1, n: 1 },
        planes: PlaneGrid { m: 2, n: 1 },
        stage_k: 48,
        delivery: CmmaDelivery::Copy,
    };
    test_matmul_strategy(
        client(),
        super::common::rect(128, 64, 96, super::common::f32_elems()),
        Tiled::Cmma(BlueprintStrategy::Forced(blueprint)).into(),
    );
}

/// A shape whose inferred plan collapses to a 1×1 partition (8×8×8 on n = 40 gives a
/// prime instruction grid along `n`), reaching the same degenerate case through the
/// selector alone.
#[test]
fn cmma_inferred_partition_1x1() {
    test_matmul_strategy(
        client(),
        super::common::rect(32, 40, 48, super::common::f32_elems()),
        Tiled::Cmma(Default::default()).into(),
    );
}

/// The TMA delivery. On a backend without TMA (Metal, wgpu, CPU) the blueprint returns
/// `Unavailable`, which the strict test policy surfaces; on CUDA it runs or fails to
/// compile, never silently degrades.
#[test]
fn cmma_tma_square_f16() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Tiled::Cmma(BlueprintStrategy::Inferred(CmmaStrategy::tma())).into(),
    );
}

/// A TMA plan whose stage exceeds the 256-per-axis box limit fails at blueprint time as a
/// clean setup error, on any backend (the plan check precedes the availability gate).
#[test]
fn cmma_tma_rejects_oversized_box() {
    use cubek_matmul::{
        definition::{AvailableVectorSizes, MatmulSetupError},
        routine::DeviceSettings,
        tiled::{
            cmma::{CmmaBlueprint, CmmaDelivery, CmmaRoutine, Partition},
            cpu_gemm::{InstructionShape, PlaneGrid},
        },
    };

    let client = client();
    // stage_n = planes.n * partition.n * instruction.n = 512 > 256.
    let blueprint = CmmaBlueprint {
        instruction: InstructionShape {
            m: 16,
            n: 16,
            k: 16,
        },
        partition: Partition { m: 2, n: 8 },
        planes: PlaneGrid { m: 2, n: 4 },
        stage_k: 16,
        delivery: CmmaDelivery::Tma,
    };
    let problem = super::common::rect(64, 1024, 64, f16_elems());
    let device_settings = DeviceSettings {
        plane_dim: client.properties().hardware.plane_size_max,
        max_cube_count: client.properties().hardware.max_cube_count,
        vector_sizes: AvailableVectorSizes::from_type_sizes(&client, 4, 4, 4)
            .pick_max()
            .unwrap(),
        client,
    };
    let strategy = BlueprintStrategy::Forced(blueprint);
    // Forced path: only `validate` runs (the acc type keys `select`'s config lookup, unused here).
    match CmmaRoutine::blueprint(
        &strategy,
        &problem,
        &device_settings,
        problem.global_dtypes.out,
    ) {
        Err(MatmulSetupError::InvalidConfig(msg)) => {
            let msg = msg.to_string();
            assert!(msg.contains("box limit"), "wrong rejection: {msg}");
        }
        Err(other) => panic!("expected a box-limit rejection, got {other:?}"),
        Ok(_) => panic!("expected a box-limit rejection, got a blueprint"),
    }
}
