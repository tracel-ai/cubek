//! Every box a packed routine cuts is a whole number of the rule's step
//! ([`MatmulStep`]), for each scheme the routines serve, in rows and stored in tiles.

use crate::kernels::MatmulRoutineProblem;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::{Axis, Space};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::MatmulStep;
use cubek_microkernel::matmul::stored::Operand;
use cubek_microkernel::matmul::stored::Packing;
use cubek_microkernel::matmul::tile_order;
use cubek_std::Hardware;
use cubek_std::devices;
use cubek_std::matmul_axes::{K, M, N, blocked};

const F16: ElemType = ElemType::Float(FloatKind::F16);

/// `value` packed along `axis` of a `[k, n]` rhs, `block` values a scale of `scale`.
fn scheme(value: QuantValue, axis: Axis, block: u8, scale: ScaleDtype) -> QuantScheme {
    let (block, store) = match axis {
        K => (vec![block, 1], QuantStore::PackedU32(1)),
        _ => (vec![1, block], QuantStore::PackedU32(0)),
    };
    QuantScheme::default()
        .with_value(value)
        .per_block(block, scale)
        .with_store(store)
}

/// `m` rows against a `4096 × 2048` rhs under `scheme`, read as `operand` states.
fn problem(m: usize, scheme: QuantScheme, operand: Operand) -> MatmulRoutineProblem {
    let packing = Packing { operand, scheme };
    MatmulRoutineProblem::packed(
        m,
        2048,
        4096,
        F16,
        MatmulOperands {
            rhs: Some(packing),
            ..MatmulOperands::all(None)
        },
    )
}

/// Every scheme the packed routines serve at `m` rows, named: NVFP4 in rows and in tile order,
/// MXFP4, MXFP8, and `q4` and `q8` along either axis.
fn problems(m: usize) -> Vec<(&'static str, MatmulRoutineProblem)> {
    let in_rows = |scheme: QuantScheme, axis: Axis| {
        let operand = Operand::of_scheme(&scheme, axis, [K, N]).expect("one axis a block");
        problem(m, scheme, operand)
    };
    let nvfp4 = tile_order::scheme();
    let words = devices::m2_pro().load_values(ElemType::UInt(UIntKind::U32));
    let tile_order = Operand::of_scheme(&nvfp4, K, [K, N])
        .expect("one axis a block")
        .stored(
            N,
            words,
            Some((
                &tile_order::values_axes(N),
                tile_order::values_storage(N, (128, 8), words).tiles(),
            )),
            None,
        )
        .expect("tile order is read");
    vec![
        ("nvfp4 rows", in_rows(nvfp4, K)),
        ("nvfp4 tile order", problem(m, nvfp4, tile_order)),
        (
            "mxfp4",
            in_rows(scheme(QuantValue::E2M1, K, 32, ScaleDtype::UE8M0), K),
        ),
        (
            "mxfp8",
            in_rows(scheme(QuantValue::E4M3, K, 32, ScaleDtype::UE8M0), K),
        ),
        (
            "q4 along k",
            in_rows(scheme(QuantValue::Q4S, K, 32, ScaleDtype::F16), K),
        ),
        (
            "q4 along n",
            in_rows(scheme(QuantValue::Q4S, N, 32, ScaleDtype::F16), N),
        ),
        (
            "q8 along k",
            in_rows(scheme(QuantValue::Q8S, K, 32, ScaleDtype::F16), K),
        ),
        (
            "q8 along n",
            in_rows(scheme(QuantValue::Q8S, N, 32, ScaleDtype::F16), N),
        ),
    ]
}

/// `problem`'s step on `hw`, contracted one value at a time.
fn step(hw: &Hardware, problem: &MatmulRoutineProblem) -> MatmulStep {
    let operands = problem
        .packings
        .map(|_, packing| packing.map_or(Operand::plain(), |packing| packing.operand));
    MatmulStep::new(hw, operands, problem.elements)
}

/// Whether `space` covers a whole number of `step`s along `M`, `N` and `K`, each over the block axis
/// and the inside axis it is split into.
fn whole_steps(step: &MatmulStep, space: &Space) -> bool {
    [M, N, K].into_iter().all(|axis| {
        let (blocks, inside) = blocked(axis);
        let extent = [blocks, inside]
            .into_iter()
            .filter(|&part| space.contains(part))
            .map(|part| space.extent(part))
            .product::<usize>();
        extent.is_multiple_of(step.along(axis))
    })
}

mod plane_shuffle {
    use super::*;
    use crate::kernels::PlaneShuffleMatmulRoutine;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// Every setting's unit box and cube box are whole steps, at one row and at eight.
    #[test]
    fn every_box_is_whole_steps() {
        let hw = devices::m2_pro();
        for m in [1, 8] {
            for (name, problem) in problems(m) {
                for settings in PlaneShuffleMatmulRoutine.packed_settings() {
                    let (blueprint, _) = PlaneShuffleMatmulRoutine
                        .prepare(&hw, &problem, &settings)
                        .unwrap_or_else(|error| panic!("{name} at {settings}: {error:?}"));
                    let partitioning = &blueprint.partitioning;
                    for space in [partitioning.leaf(), partitioning.cube_box()] {
                        assert!(
                            whole_steps(&step(&hw, &problem), &space),
                            "{name} at {settings}: {space:?}"
                        );
                    }
                }
            }
        }
    }
}

mod cube_shuffle {
    use super::*;
    use crate::kernels::CubeShuffleMatmulRoutine;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// Every setting that prepares cuts whole steps, at one row and at eight. It declines what
    /// leaves its planes fewer than two turns of `K` each, so only some prepare.
    #[test]
    fn every_box_is_whole_steps() {
        let hw = devices::m2_pro();
        let mut prepared = 0;
        for m in [1, 8] {
            for (name, problem) in problems(m) {
                let routine = CubeShuffleMatmulRoutine;
                for settings in routine.packed_settings(&hw) {
                    let Ok((blueprint, _)) = routine.prepare(&hw, &problem, &settings) else {
                        continue;
                    };
                    prepared += 1;
                    let partitioning = &blueprint.partitioning;
                    for space in [partitioning.leaf(), partitioning.cube_box()] {
                        assert!(
                            whole_steps(&step(&hw, &problem), &space),
                            "{name} at {settings}: {space:?}"
                        );
                    }
                }
            }
        }
        assert!(prepared > 0);
    }
}

mod cube_staged {
    use super::*;
    use crate::kernels::{CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings};
    use cubecl::features::MmaConfig;
    use cubecl::tile::StageStorage;
    use cubecl::tile::launch::Refusal;
    use cubecl::tile::stage::RowChunks;
    use cubek_microkernel::matmul::MatmulDim;
    use cubek_microkernel::matmul::dense::cube::CubeStagedMicrokernel;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentTransport;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// Every setting's cube box is whole steps under the instruction its planes run, at a
    /// prompt's rows, where its stages fit the device's shared memory: a stage is whole scale
    /// blocks of `K`, as many as the step covers. An rhs whose scales do not group `K` is
    /// declined by name, and tile order prepares.
    #[test]
    fn every_box_is_whole_steps() {
        // The M2 as Metal reports it: its simdgroup matrix, `8 × 8 × 8` over `f16` into `f32`.
        let hw = devices::offering_cmma(
            devices::m2_pro(),
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: ElemType::Float(FloatKind::F32),
                m: 8,
                n: 8,
                k: 8,
            }],
        );
        let mut tile_order_prepared = 0;
        for (name, problem) in problems(512) {
            for settings in CubeStagedMatmulRoutine.packed_settings(&hw) {
                let prepared = CubeStagedMatmulRoutine.prepare(&hw, &problem, &settings);
                let blueprint = match prepared {
                    Ok((blueprint, _)) => blueprint,
                    Err(Refusal::SharedMemoryPastDevice { .. }) => continue,
                    Err(error) if name.ends_with("along n") => {
                        assert_eq!(error, CubeStagedMicrokernel::UNGROUPED_K, "{name}");
                        continue;
                    }
                    Err(error) => panic!("{name} at {settings}: {error:?}"),
                };
                if name == "nvfp4 tile order" {
                    tile_order_prepared += 1;
                }
                let [(_, m), (_, n), (_, k)] = blueprint.microkernel.plane.instruction.shape();
                let step = step(&hw, &problem).under(Space::new(&[(M, m), (N, n), (K, k)]));
                let cube = blueprint.partitioning.cube_box();
                assert!(whole_steps(&step, &cube), "{name} at {settings}: {cube:?}");
            }
        }
        assert!(tile_order_prepared > 0, "no setting prepared tile order");
    }

    /// A swizzled stage is read a chunk a unit, so the rhs decoding into one binds a line no
    /// wider than a chunk; a padded one takes a whole instruction step of `K` a line, so one unit
    /// decodes a whole scale block. Every packed box through each transport, on a device offering
    /// `f16` to `cmma` and to `mma`.
    #[test]
    fn a_packed_rhs_decoding_into_a_swizzled_stage_binds_a_chunk_a_line() {
        let f16_into_f32 = |m, n, k| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: ElemType::Float(FloatKind::F32),
            m,
            n,
            k,
        };
        let hw = devices::offering_mma(
            devices::offering_cmma(devices::strix_halo(), [f16_into_f32(16, 16, 16)]),
            [f16_into_f32(16, 8, 16)],
        );
        let (_, problem) = problems(512)
            .into_iter()
            .find(|(name, _)| *name == "nvfp4 tile order")
            .expect("the problems hold tile order");
        let mut laid = Vec::new();
        let settings = CubeStagedMatmulRoutine.packed_settings(&hw);
        let through_mma = settings
            .iter()
            .map(|&settings| CubeStagedMatmulRoutineSettings {
                microkernel: settings.microkernel.through(PlaneFragmentTransport::Mma),
                ..settings
            });
        for settings in settings.iter().copied().chain(through_mma) {
            let Ok((blueprint, _)) = CubeStagedMatmulRoutine.prepare(&hw, &problem, &settings)
            else {
                continue;
            };
            let StageStorage::Tiled { chunks, .. } = blueprint.shared_memory else {
                panic!("{settings}: a matrix instruction's stage is tiled");
            };
            let step = blueprint.microkernel.plane.leaf_extent(MatmulDim::K);
            let rhs = blueprint.widths.rhs;
            match chunks {
                RowChunks::Swizzled => {
                    assert!(
                        rhs * F16.size() <= RowChunks::CHUNK_BYTES,
                        "{settings}: {rhs}"
                    )
                }
                RowChunks::Padded | RowChunks::InOrder => assert_eq!(rhs, step, "{settings}"),
            }
            laid.push(chunks);
        }
        assert!(laid.contains(&RowChunks::Swizzled), "no setting swizzled");
        assert!(laid.contains(&RowChunks::Padded), "no setting padded");
    }
}

mod packed_lhs {
    use super::*;
    use crate::kernels::select::packed::nvfp4_scheme;
    use crate::kernels::{
        CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, DequantizedMatmulRoutine,
        PlaneShuffleMatmulRoutine,
    };
    use cubecl::features::MmaConfig;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// `m` rows of an `[m, k]` input packed in NVFP4 along `k`, against a float `[k, n]` one.
    fn packed_lhs(m: usize) -> MatmulRoutineProblem {
        let scheme = nvfp4_scheme();
        let packing = Packing {
            operand: Operand::of_scheme(&scheme, K, [M, K]).expect("one axis a block"),
            scheme,
        };
        MatmulRoutineProblem::packed(
            m,
            2048,
            4096,
            F16,
            MatmulOperands {
                lhs: Some(packing),
                ..MatmulOperands::all(None)
            },
        )
    }

    /// A packed `[m, k]` input against a float `[k, n]` one is declined by the routines that
    /// stream the `[k, n]` input, and served by the routine that decodes every packed input
    /// first: the default a product with nothing chosen falls back to.
    #[test]
    fn a_packed_lhs_is_decoded_where_the_streaming_routines_decline_it() {
        let hw = devices::offering_cmma(
            devices::m2_pro(),
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: ElemType::Float(FloatKind::F32),
                m: 8,
                n: 8,
                k: 8,
            }],
        );
        for m in [1, 8] {
            let problem = packed_lhs(m);
            for settings in PlaneShuffleMatmulRoutine
                .settings(&hw)
                .into_iter()
                .chain(PlaneShuffleMatmulRoutine.packed_settings())
            {
                assert!(
                    PlaneShuffleMatmulRoutine
                        .prepare(&hw, &problem, &settings)
                        .is_err(),
                    "plane shuffle at {settings}, {m} rows"
                );
            }
            for settings in CubeShuffleMatmulRoutine.packed_settings(&hw) {
                assert!(
                    CubeShuffleMatmulRoutine
                        .prepare(&hw, &problem, &settings)
                        .is_err(),
                    "cube shuffle at {settings}, {m} rows"
                );
            }
        }
        let problem = packed_lhs(512);
        let decoded = DequantizedMatmulRoutine {
            dense: CubeStagedMatmulRoutine,
        };
        assert!(
            decoded
                .settings(&hw)
                .iter()
                .any(|settings| decoded.prepare(&hw, &problem, settings).is_ok()),
            "no setting decodes the packed lhs and runs the float product"
        );
    }
}
