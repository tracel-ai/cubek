//! Tests for the attention cube staged routine, over plane fragments and plane unit groups: its
//! settings and what each prepares.

mod fragment {
    use super::super::*;
    use crate::kernels::AttentionRoutineProblem;
    use cubecl::tile::launch::Refusal;
    use cubek_microkernel::attention::QueryOrder;
    use cubek_microkernel::attention::cube::CubeAttentionMicrokernelSettings;
    use cubek_microkernel::attention::plane::PlaneFragmentAttentionMicrokernel;
    use cubek_microkernel::attention::plane::PlaneFragmentAttentionMicrokernelSettings;
    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    use core::num::NonZeroUsize;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::stage::RowChunks;
    use cubek_microkernel::attention::CubeAttentionStageLayout;
    use cubek_microkernel::attention::plane::InstructionShape;
    use cubek_microkernel::attention::plane::{
        ROW_STATE_HELD_BY_UNIT, SWIZZLED_STAGE_UNREAD, ScoreHolder,
    };
    use cubek_microkernel::attention::{AttentionMicrokernel, CubePlaneMicrokernel};
    use cubek_microkernel::attention::{G, Q, S};
    use cubek_microkernel::staging::StageRing;

    const ROUTINE: CubeStagedAttentionRoutine<PlaneFragmentAttentionMicrokernel> =
        CubeStagedAttentionRoutine::new();
    type Settings = CubeStagedAttentionRoutineSettings<PlaneFragmentAttentionMicrokernel>;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the square instruction of `edge` for `f16` and for `f32`, summed in `f32`.
    fn offering(hw: Hardware, edge: u32) -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: edge,
            n: edge,
            k: edge,
        };
        devices::offering_cmma(hw, [square(F16), square(F32)])
    }

    /// A decode step, a prompt whose queries divide nothing, a wide group over a long cache, and
    /// a head dim of 128 in `f32`, the widest the leads are held to.
    fn problems() -> [AttentionRoutineProblem; 4] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 2048), (128, 128), F16),
            AttentionRoutineProblem::dense((2, 8, 2), (37, 300), (64, 64), F16),
            AttentionRoutineProblem::dense((1, 16, 1), (512, 4096), (128, 128), F16),
            AttentionRoutineProblem::dense((1, 8, 2), (64, 256), (128, 128), F32),
        ]
    }

    /// Every setting prepares where the device offers a square instruction and a cube holds a plane
    /// for every head; one other than the default may ask for more than a cube holds, or for a
    /// layout its planes do not read, and says so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [
            offering(devices::m2_pro(), 8),
            offering(devices::h100(), 16),
        ] {
            for (position, settings) in ROUTINE.settings(&hw).iter().enumerate() {
                for problem in problems() {
                    let prepared = ROUTINE.prepare(&hw, &problem, settings);
                    let declined = matches!(
                        prepared,
                        Err(Refusal::SharedMemoryPastDevice { .. } | Refusal::CubePastDevice { .. })
                    ) || prepared.as_ref().err() == Some(&SWIZZLED_STAGE_UNREAD);
                    assert!(
                        prepared.is_ok() || (position > 0 && declined),
                        "{settings} on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// `hw` offering what an NVIDIA GPU does through cubecl's fragments: `16×16×16`, `32×8×16`
    /// and `8×32×16` for `f16`, and `16×16×8` for tf32 alone, each summed in `f32`.
    fn offering_nvidia(hw: Hardware) -> Hardware {
        let shape = |element, (m, n, k)| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m,
            n,
            k,
        };
        devices::offering_cmma(
            hw,
            [
                shape(F16, (16, 16, 16)),
                shape(F16, (32, 8, 16)),
                shape(F16, (8, 32, 16)),
                shape(ElemType::Float(FloatKind::TF32), (16, 16, 8)),
            ],
        )
    }

    /// On an NVIDIA GPU, `f16` runs on the square `16×16×16` unless a setting asks otherwise.
    #[test]
    fn f16_runs_on_a_square_instruction_by_default() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, f16, _] = problems();
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &f16, &Settings::DEFAULT)
            .expect("f16 prepares on an NVIDIA GPU's shapes");
        let [(_, m), (_, n), (_, k)] = blueprint.microkernel.plane.score.instruction.shape();
        assert_eq!((m, n, k), (16, 16, 16));
    }

    /// On an NVIDIA GPU, `f16` asked for a rectangular instruction takes the first its shapes
    /// rank, `8×32×16`: the score's fragments are `8` queries by `32` keys over head steps of
    /// `16`, the value's `8` queries by `32` of the value dim over key steps of `16`, two to the
    /// score's one.
    #[test]
    fn a_rectangular_instruction_cuts_each_product_its_own_way() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, f16, _] = problems();
        let lead = Settings::DEFAULT;
        let rectangular = Settings {
            microkernel: CubeAttentionMicrokernelSettings {
                plane: PlaneFragmentAttentionMicrokernelSettings {
                    shape: InstructionShape::Rectangular,
                    ..lead.microkernel.plane
                },
                ..lead.microkernel
            },
            ..lead
        };
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &f16, &rectangular)
            .expect("f16 prepares on an NVIDIA GPU's shapes");
        let plane = blueprint.microkernel.plane;
        assert_eq!(
            plane.score.instruction.shape(),
            plane.value.instruction.shape()
        );
        assert_eq!(
            (plane.score.tile_extent(S), plane.score.settings.depth.get()),
            (32, 128 / 16)
        );
        assert_eq!(
            (plane.value.tile_extent(S), plane.value.settings.depth.get()),
            (32, 2)
        );
    }

    /// On an NVIDIA GPU, which offers `f32` no instruction of its own, `f32` is read as tf32 and
    /// runs on its `16×16×8`: the score's fragments are `16` queries by `16` keys over head steps
    /// of `8`, the value's `16` of the value dim over key steps of `8`, two to the score's one.
    #[test]
    fn f32_runs_on_tf32_where_the_device_offers_nothing_exact() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, _, f32] = problems();
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &f32, &Settings::DEFAULT)
            .expect("f32 prepares on tf32's 16×16×8");
        let plane = blueprint.microkernel.plane;
        assert_eq!(plane.read, ElemType::Float(FloatKind::TF32));
        assert_eq!(
            (plane.score.tile_extent(S), plane.score.settings.depth.get()),
            (16, 128 / 8)
        );
        assert_eq!(
            (plane.value.tile_extent(S), plane.value.settings.depth.get()),
            (16, 2)
        );
    }

    /// Read as tf32, `f32` operands are bound as words, which the register prefetch does not
    /// carry: a ring asked to prefetch through registers prefetches into its slots.
    #[test]
    fn tf32_operands_prefetch_into_their_slots() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, _, f32] = problems();
        let lead = Settings::DEFAULT;
        let through_registers = Settings {
            microkernel: CubeAttentionMicrokernelSettings {
                ring: StageRing::SINGLE_THROUGH_REGISTERS,
                ..lead.microkernel
            },
            ..lead
        };
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &f32, &through_registers)
            .expect("f32 prepares on tf32");
        assert_eq!(blueprint.microkernel.ring, StageRing::SINGLE);
    }

    /// A padded stage costs one chunk a row of its keys and values, and a blocked one lays the same
    /// cells out at no cost: what the cube counts, so a layout that does not fit the device
    /// declines instead of overflowing. Padded blocks of the instruction's tiles cost nothing either
    /// here: a tile's row of eight `f16` is one chunk, which cubek keeps in order.
    #[test]
    fn a_padded_stage_costs_a_chunk_a_row() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, _, problem, _] = problems();
        let item = problem.work_item();
        let prepared = |layout| {
            let settings = Settings {
                microkernel: CubeAttentionMicrokernelSettings {
                    layout,
                    ..Settings::DEFAULT.microkernel
                },
                ..Settings::DEFAULT
            };
            let (blueprint, _) = ROUTINE
                .prepare(&hw, &problem, &settings)
                .expect("the lead's box prepares on the M2 in every layout");
            blueprint
        };
        let bytes = |layout| {
            let blueprint = prepared(layout);
            blueprint
                .microkernel
                .shared_memory(&hw, &item, &blueprint.shared_memory)
        };
        let microkernel = prepared(CubeAttentionStageLayout::Strided).microkernel;
        let stage = item.stage(microkernel.seq_kv(), microkernel.key_order());
        let whole_rows: usize = stage
            .iter()
            .map(|operand| operand.cells() / operand.space.extent(operand.innermost))
            .sum();
        let chunk = RowChunks::CHUNK_BYTES.max(hw.vector_ceiling(F16) * F16.size());
        let strided = bytes(CubeAttentionStageLayout::Strided);
        assert_eq!(
            bytes(CubeAttentionStageLayout::PaddedRows),
            strided + whole_rows * chunk
        );
        assert_eq!(bytes(CubeAttentionStageLayout::Blocks), strided);
        assert_eq!(
            microkernel.plane.out_edge() * F16.size(),
            RowChunks::CHUNK_BYTES
        );
        assert_eq!(bytes(CubeAttentionStageLayout::PaddedBlocks), strided);
    }

    /// A cube's box is its heads over its queries, a plane each, and its stage is that box against
    /// a stage of keys: the keys are staged once for every head the cube holds.
    #[test]
    fn a_cube_stages_its_keys_once_for_every_head() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, problem, _, _] = problems();
        let (blueprint, grid) = ROUTINE
            .prepare(&hw, &problem, &Settings::DEFAULT)
            .expect("the default prepares on the M2");
        let (cube, stage) = (
            blueprint.partitioning.cube_box(),
            blueprint.partitioning.stage(),
        );
        let microkernel = blueprint.microkernel;
        let heads = microkernel.heads;
        assert_eq!(heads, problem.group.min(4));
        assert_eq!(cube.extent(G), heads);
        assert_eq!(stage.extent(G), heads);
        assert_eq!(stage.extent(S), microkernel.plane.seq_kv());
        assert_eq!(grid.planes().count, heads);
    }

    /// Planes splitting the same rows' keys merge their softmax states row by row, so they read
    /// the score through their windows, where every unit holds every row's; one plane along the
    /// keys keeps it in its fragments.
    #[test]
    fn planes_splitting_the_keys_read_the_score_through_their_windows() {
        let manual = |k| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k,
        };
        let hw = devices::offering_mma(offering(devices::h100(), 16), [manual(16), manual(8)]);
        let holder = |settings: CubeAttentionMicrokernelSettings<_>| {
            let (blueprint, _) = ROUTINE
                .prepare(
                    &hw,
                    &problems()[2],
                    &Settings {
                        microkernel: settings,
                        ..Settings::DEFAULT
                    },
                )
                .expect("prepares where both transports are offered");
            blueprint.microkernel.plane.score_holder()
        };
        let settings = ROUTINE.settings(&hw);
        let lead = settings[0].microkernel;
        assert_eq!(holder(lead), ScoreHolder::Fragments);
        for split in settings
            .iter()
            .filter(|settings| settings.microkernel.planes_along_keys.get() > 1)
        {
            assert_eq!(holder(split.microkernel), ScoreHolder::Window, "{split}");
        }
    }

    /// A score kept in fragments leaves each unit its own rows' state, which no other plane's
    /// merges into: planes splitting the keys decline it where nothing else is offered.
    #[test]
    fn planes_splitting_the_keys_decline_a_score_kept_in_fragments() {
        let manual = |k| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k,
        };
        let hw = devices::offering_mma(
            devices::offering_cmma(devices::h100(), []),
            [manual(16), manual(8)],
        );
        let split = CubeAttentionMicrokernelSettings {
            heads: NonZeroUsize::new(1).unwrap(),
            planes_along_keys: NonZeroUsize::new(2).unwrap(),
            ..Settings::DEFAULT.microkernel
        };
        let prepared = ROUTINE.prepare(
            &hw,
            &problems()[2],
            &Settings {
                microkernel: split,
                ..Settings::DEFAULT
            },
        );
        assert_eq!(
            prepared.err(),
            Some(ROW_STATE_HELD_BY_UNIT),
            "planes splitting the keys decline a score kept in fragments"
        );
    }

    /// The lead takes the boxes of queries in order, and the one setting listed last first is
    /// its default reversed: a table may race the order, and nothing takes it unasked.
    #[test]
    fn only_the_listed_setting_takes_the_queries_last_first() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, _, problem, _] = problems();
        let reverses_queries = |settings: &Settings| {
            let (blueprint, _) = ROUTINE
                .prepare(&hw, &problem, settings)
                .expect("the lead's box prepares on the M2 in either order");
            blueprint
                .partitioning
                .as_tile()
                .levels()
                .iter()
                .any(|level| level.is_reversed(Q))
        };
        let settings = ROUTINE.settings(&hw);
        let last_first: Vec<_> = settings
            .iter()
            .filter(|settings| settings.queries == QueryOrder::LastFirst)
            .collect();
        assert_eq!(
            last_first,
            [&Settings {
                queries: QueryOrder::LastFirst,
                ..Settings::DEFAULT
            }]
        );
        assert!(!reverses_queries(&settings[0]));
        assert!(reverses_queries(last_first[0]));
    }
}

mod unit_group {
    use super::super::*;
    use crate::kernels::AttentionRoutineProblem;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::launch::Refusal;
    use cubek_microkernel::attention::plane::PlaneUnitGroupAttentionMicrokernel;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    const ROUTINE: CubeStagedAttentionRoutine<PlaneUnitGroupAttentionMicrokernel> =
        CubeStagedAttentionRoutine::new();

    const BF16: ElemType = ElemType::Float(FloatKind::BF16);
    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// A decode step, a prompt whose queries divide nothing, and a long prompt in `bf16`.
    fn problems() -> [AttentionRoutineProblem; 3] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 2048), (128, 128), F16),
            AttentionRoutineProblem::dense((2, 8, 2), (37, 300), (64, 64), F16),
            AttentionRoutineProblem::dense((1, 16, 2), (512, 512), (128, 128), BF16),
        ]
    }

    /// Every setting prepares where the device commits to a plane width, with no matrix
    /// instruction at all; one other than the default may ask for more than a cube holds, and
    /// says so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [devices::m2_pro(), devices::h100()] {
            for (position, settings) in ROUTINE.settings(&hw).iter().enumerate() {
                for problem in problems() {
                    let prepared = ROUTINE.prepare(&hw, &problem, settings);
                    let outgrown = matches!(
                        prepared,
                        Err(Refusal::SharedMemoryPastDevice { .. } | Refusal::CubePastDevice { .. })
                    );
                    assert!(
                        prepared.is_ok() || (position > 0 && outgrown),
                        "{settings} on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// Every setting declines where the device commits to no plane width, rather than launching
    /// on lanes it would misread.
    #[test]
    fn every_setting_declines_without_a_plane_width() {
        for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
            for settings in ROUTINE.settings(&devices::m2_pro()) {
                for problem in problems() {
                    let declined = ROUTINE.prepare(&hw, &problem, &settings);
                    assert!(declined.is_err(), "{settings} on {hw:?}: {declined:?}");
                }
            }
        }
    }
}
