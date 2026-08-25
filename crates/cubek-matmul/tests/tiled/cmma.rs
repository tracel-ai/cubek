//! Inferred-blueprint smoke tests for the tile-DSL cmma routine: the port of the
//! multi-level simple cyclic cmma matmul.

use cubek_matmul::{
    routine::BlueprintStrategy,
    tiled::{Strategy as Tiled, cmma::CmmaStrategy},
};

use crate::harness::{client, f16_elems, f32_elems, rect, square, test_matmul_strategy};

#[test]
fn cmma_square_f32() {
    test_matmul_strategy(
        client(),
        square(256, f32_elems()),
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
        rect(64, 128, 32, f32_elems()),
        Tiled::Cmma(Default::default()).into(),
    );
}

#[test]
fn cmma_batched_f32() {
    use cubecl::{ir::AddressType, zspace::shape};
    use cubek_matmul::definition::MatmulProblem;
    use cubek_std::MatrixLayout;

    let elems = f32_elems();
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
        rect(128, 64, 96, f32_elems()),
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
        rect(32, 40, 48, f32_elems()),
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
    let problem = rect(64, 1024, 64, f16_elems());
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

/// A caller asking for an input register type of its own (tf32 fragments off an f32 tensor)
/// is asking for a cast this routine does not emit. It is rejected at setup rather than run
/// at the global type behind the caller's back.
#[test]
fn cmma_rejects_input_register_type() {
    use cubecl::{
        ir::{ElemType, FloatKind},
        prelude::*,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulSetupError},
        tiled::cmma::launch_ref,
    };
    use cubek_std::{InputBinding, MatrixLayout};
    use cubek_test_utils::TestInput;

    let client = client();
    let f32t = f32::elem_type_native();
    let tensor = |seed| {
        TestInput::builder(client.clone(), vec![64, 64])
            .dtype(f32t)
            .layout(MatrixLayout::RowMajor)
            .uniform(seed, -1., 1.)
            .generate_without_host_data()
    };
    let (lhs, rhs, out) = (tensor(1234), tensor(5678), tensor(4242));

    // Everything f32 but the lhs fragment: the one field the kernel has no `EL` for.
    let mut dtypes = MatmulElems::from_single_dtype(f32t);
    dtypes.lhs_register = ElemType::Float(FloatKind::TF32);

    match launch_ref(
        &client,
        InputBinding::Normal(lhs.binding(), f32t),
        InputBinding::Normal(rhs.binding(), f32t),
        out.binding(),
        &Default::default(),
        &dtypes,
    ) {
        Err(MatmulSetupError::InvalidConfig(msg)) => {
            let msg = msg.to_string();
            assert!(
                msg.contains("Lhs") && msg.contains("TF32"),
                "wrong rejection: {msg}"
            );
        }
        other => panic!("expected a type rejection, got {other:?}"),
    }
}
