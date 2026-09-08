//! Inferred-blueprint smoke tests for the tile-DSL cmma routine: the port of the
//! multi-level simple cyclic cmma matmul.

use cubecl::prelude::*;
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
        buffering: 2,
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
            cmma::{CmmaBlueprint, CmmaDelivery, CmmaRoutine, Partition, StoredTiles},
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
        buffering: 2,
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
        StoredTiles::default(),
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

/// The weight packed at load into blocks of exactly the plan's stage and launched under the
/// Tiled delivery beside a row-major activation: the same product as the row-major run, read
/// one contiguous block per stage. The pack is a copy through the two tiles' views, so the
/// block layout the kernel reads is the one the tile engine itself describes.
#[test]
fn cmma_storage_tiled_weight_f16() {
    use cubecl::frontend::Scalar;
    storage_tiled_weight(half::f16::elem_type_native());
}

#[test]
fn cmma_storage_tiled_weight_f32() {
    use cubecl::frontend::Scalar;
    storage_tiled_weight(f32::elem_type_native());
}

fn storage_tiled_weight(dtype: ElemType) {
    use cubecl::ir::FloatKind;
    use cubek_matmul::{
        definition::{AvailableVectorSizes, MatmulElems, MatmulSetupError},
        routine::DeviceSettings,
        tiled::{
            cmma::{CmmaRoutine, StoredTiles, launch_ref},
            pack::pack,
        },
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

    use crate::harness::assert_result;

    let client = client();
    let (m, n, k) = (128, 256, 384);
    let dtypes = MatmulElems::from_single_dtype(dtype);
    let problem = rect(m, n, k, dtypes.as_global_elems());

    let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(dtype)
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();
    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(dtype)
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(dtype)
        .uniform(4242, 10., 100.)
        .generate_without_host_data();

    let mut elems = dtypes.clone();
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        let launch = || -> Result<(), MatmulSetupError> {
            // The plan the selector picks for this problem, under the Tiled delivery: its stage
            // is the block the weight is packed to.
            let acc = match dtype {
                ElemType::Float(FloatKind::F16 | FloatKind::BF16) => f32::elem_type_native(),
                other => other,
            };
            let sz = dtype.size();
            let device_settings = DeviceSettings {
                client: c.clone(),
                plane_dim: c.properties().hardware.plane_size_max,
                vector_sizes: AvailableVectorSizes::from_type_sizes(c, sz, sz, sz).pick_max()?,
                max_cube_count: c.properties().hardware.max_cube_count,
            };
            let blueprint = CmmaRoutine::blueprint(
                &BlueprintStrategy::Inferred(CmmaStrategy::tiled()),
                &problem,
                &device_settings,
                acc,
                StoredTiles::default(),
            )?;
            let (_, stage_n) = blueprint.stage();
            let stage_k = blueprint.stage_k;

            // The weight, packed to the plan's stage.
            let packed = pack(c, rhs.clone().binding(), dtype, (stage_k, stage_n))?;

            launch_ref(
                c,
                InputBinding::Normal(lhs.clone().binding(), dtype),
                InputBinding::Normal(packed.binding(), dtype),
                out.clone().binding(),
                &BlueprintStrategy::Forced(blueprint),
                &elems,
            )
        };
        launch().into()
    });
    elems.acc_global = dtype;

    match outcome {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, out, dtypes).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}

/// A weight is packed once and read at every `m`. Its storage tile names the stage: an inferred
/// plan stages to it, whatever this `m` would have chosen, and moves it under the Tiled delivery.
#[test]
fn cmma_packed_weight_names_the_stage_across_m() {
    use cubek_matmul::{
        definition::{AvailableVectorSizes, MatmulElems, MatmulSetupError},
        routine::DeviceSettings,
        tiled::{
            cmma::{CmmaDelivery, CmmaRoutine, StoredTiles, launch_ref},
            pack::pack,
        },
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

    use crate::harness::assert_result;

    let client = client();
    let dtype = f32::elem_type_native();
    let (n, k) = (512, 256);
    let dtypes = MatmulElems::from_single_dtype(dtype);
    let sz = dtype.size();
    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: client.properties().hardware.plane_size_max,
        vector_sizes: AvailableVectorSizes::from_type_sizes(&client, sz, sz, sz)
            .pick_max()
            .unwrap(),
        max_cube_count: client.properties().hardware.max_cube_count,
    };

    // A storage tile the selector would not have picked on its own: twice its stage depth and
    // stage width for an f32 weight.
    let tile = (32, 64);
    let (rhs, rhs_data) = TestInput::builder(
        client.clone(),
        rect(64, n, k, dtypes.as_global_elems()).rhs_shape,
    )
    .dtype(dtype)
    .uniform(5678, -1., 1.)
    .generate_with_f32_host_data();
    let packed = pack(&client, rhs.binding(), dtype, tile).unwrap();

    for m in [64, 512] {
        let problem = rect(m, n, k, dtypes.as_global_elems());
        let free = CmmaRoutine::blueprint(
            &BlueprintStrategy::Inferred(CmmaStrategy::default()),
            &problem,
            &device_settings,
            dtype,
            StoredTiles::default(),
        )
        .unwrap();
        let held = CmmaRoutine::blueprint(
            &BlueprintStrategy::Inferred(CmmaStrategy::default()),
            &problem,
            &device_settings,
            dtype,
            StoredTiles {
                lhs: None,
                rhs: Some(tile),
            },
        )
        .unwrap();
        assert_eq!(free.delivery, CmmaDelivery::Copy);
        assert_eq!(held.delivery, CmmaDelivery::Tiled);
        assert_eq!((held.stage_k, held.stage().1), tile, "at m = {m}");
        assert_ne!((free.stage_k, free.stage().1), tile, "at m = {m}");

        let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
            .dtype(dtype)
            .uniform(1234 + m as u64, -1., 1.)
            .generate_with_f32_host_data();
        let out = TestInput::builder(client.clone(), problem.out_shape.clone())
            .dtype(dtype)
            .uniform(4242, 10., 100.)
            .generate_without_host_data();
        let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
            let launch = || -> Result<(), MatmulSetupError> {
                launch_ref(
                    c,
                    InputBinding::Normal(lhs.clone().binding(), dtype),
                    InputBinding::Normal(packed.clone().binding(), dtype),
                    out.clone().binding(),
                    &BlueprintStrategy::Inferred(CmmaStrategy::default()),
                    &dtypes,
                )
            };
            launch().into()
        });
        match outcome {
            ExecutionOutcome::Executed => {
                assert_result(&lhs_data, &rhs_data, &problem, &client, out, dtypes.clone())
                    .as_test_outcome()
            }
            ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
        }
        .enforce();
    }
}

/// Both operands stored, each naming `stage_k` through its tile: they must agree.
#[test]
fn cmma_refuses_stored_operands_that_disagree_on_k() {
    use cubek_matmul::{
        definition::{AvailableVectorSizes, MatmulElems, MatmulSetupError},
        routine::DeviceSettings,
        tiled::cmma::{CmmaRoutine, StoredTiles},
    };

    let client = client();
    let dtype = f32::elem_type_native();
    let dtypes = MatmulElems::from_single_dtype(dtype);
    let problem = rect(64, 512, 256, dtypes.as_global_elems());
    let sz = dtype.size();
    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: client.properties().hardware.plane_size_max,
        vector_sizes: AvailableVectorSizes::from_type_sizes(&client, sz, sz, sz)
            .pick_max()
            .unwrap(),
        max_cube_count: client.properties().hardware.max_cube_count,
    };
    match CmmaRoutine::blueprint(
        &BlueprintStrategy::Inferred(CmmaStrategy::default()),
        &problem,
        &device_settings,
        dtype,
        StoredTiles {
            lhs: Some((16, 32)),
            rhs: Some((16, 64)),
        },
    ) {
        Err(MatmulSetupError::InvalidConfig(msg)) => {
            let msg = msg.to_string();
            assert!(msg.contains("same depth"), "wrong rejection: {msg}")
        }
        Err(other) => panic!("expected a stage_k disagreement, got {other:?}"),
        Ok(_) => panic!("expected a stage_k disagreement, got a blueprint"),
    }
}
