//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubek_matmul::routines::{BlueprintStrategy, cmma::CmmaStrategy};
use cubek_matmul::strategy::Strategy;

use super::common::{client, f16_elems, square};
use crate::matmul::test_matmul_strategy;

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleCyclicCmma(Default::default()),
    );
}

#[test]
fn simple_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleCyclicMma(Default::default()),
    );
}

#[test]
fn simple_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleStridedCmma(Default::default()),
    );
}

#[test]
fn simple_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleStridedMma(Default::default()),
    );
}

#[test]
fn simple_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTilewiseCmma(Default::default()),
    );
}

#[test]
fn simple_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleTilewiseMma(Default::default()),
    );
}

#[test]
fn simple_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncStridedCmma(Default::default()),
    );
}

#[test]
fn simple_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncStridedMma(Default::default()),
    );
}

#[test]
fn simple_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncCyclicCmma(Default::default()),
    );
}

#[test]
fn simple_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SimpleAsyncCyclicMma(Default::default()),
    );
}

#[test]
fn double_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleCyclicCmma(Default::default()),
    );
}

#[test]
fn double_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleCyclicMma(Default::default()),
    );
}

#[test]
fn double_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTilewiseCmma(Default::default()),
    );
}

#[test]
fn double_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleTilewiseMma(Default::default()),
    );
}

#[test]
fn double_hybrid_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleHybridCmma(Default::default()),
    );
}

#[test]
fn double_hybrid_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleHybridMma(Default::default()),
    );
}

#[test]
fn double_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncCyclicCmma(Default::default()),
    );
}

#[test]
fn double_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncCyclicMma(Default::default()),
    );
}

#[test]
fn double_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncStridedCmma(Default::default()),
    );
}

#[test]
fn double_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::DoubleAsyncStridedMma(Default::default()),
    );
}

#[test]
fn specialized_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedCyclicCmma(Default::default()),
    );
}

#[test]
fn specialized_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedCyclicMma(Default::default()),
    );
}

#[test]
fn specialized_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedStridedCmma(Default::default()),
    );
}

#[test]
fn specialized_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::SpecializedStridedMma(Default::default()),
    );
}

#[test]
fn ordered_double_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::OrderedDoubleCmma(Default::default()),
    );
}

#[test]
fn ordered_double_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::OrderedDoubleMma(Default::default()),
    );
}

// ---- the tile-DSL port of the simple cyclic cmma matmul --------------------------------

#[test]
fn cmma_square_f32() {
    test_matmul_strategy(
        client(),
        square(256, super::common::f32_elems()),
        Strategy::Cmma(Default::default()),
    );
}

#[test]
fn cmma_square_f16() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::Cmma(Default::default()),
    );
}

#[test]
fn cmma_rect_f32() {
    test_matmul_strategy(
        client(),
        super::common::rect(64, 128, 32, super::common::f32_elems()),
        Strategy::Cmma(Default::default()),
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
    test_matmul_strategy(client(), problem, Strategy::Cmma(Default::default()));
}

/// The TMA delivery. On a backend without TMA (Metal, wgpu, CPU) the blueprint returns
/// `Unavailable`, which the strict test policy surfaces; on CUDA it runs or fails to
/// compile, never silently degrades.
#[test]
fn cmma_tma_square_f16() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        Strategy::Cmma(BlueprintStrategy::Inferred(CmmaStrategy::tma())),
    );
}

/// A TMA plan whose stage exceeds the 256-per-axis box limit fails at blueprint time as a
/// clean setup error, on any backend (the plan check precedes the availability gate).
#[test]
fn cmma_tma_rejects_oversized_box() {
    use cubek_matmul::definition::{AvailableVectorSizes, MatmulSetupError};
    use cubek_matmul::routines::{
        DeviceSettings,
        cmma::{CmmaBlueprint, CmmaRoutine, Partition},
        cpu_gemm::{Instruction, PlaneGrid},
    };
    use cubek_tile::Delivery;

    let client = client();
    // stage_n = planes.n * partition.n * instruction.n = 512 > 256.
    let blueprint = CmmaBlueprint {
        instruction: Instruction {
            m: 16,
            n: 16,
            k: 16,
        },
        partition: Partition { m: 2, n: 8 },
        planes: PlaneGrid { m: 2, n: 4 },
        stage_k: 16,
        delivery: Delivery::Tma,
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
