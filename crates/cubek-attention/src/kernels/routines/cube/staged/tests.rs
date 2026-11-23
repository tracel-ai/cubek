//! Tests for the attention cube staged routine over plane fragments: its settings and what each
//! prepares.

mod fragment {
    use super::super::*;
    use crate::kernels::{AttentionRoutineProblem, QueryOrder};
    use cubecl::tile::launch::{Decline, Delivery};
    use cubek_microkernel::attention::AttentionRead;
    use cubek_microkernel::attention::cube::CubeAttentionMicrokernelSettings;
    use cubek_microkernel::matmul::cube::CubeStagedDelivery;

    use cubek_microkernel::attention::plane::BLOCK_ROW_BYTES;
    use cubek_std::routine::Routine;
    use cubek_std::{Hardware, devices};

    use core::num::NonZeroUsize;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::stage::RowChunks;
    use cubek_microkernel::attention::cube::{CubeAttentionPlane, CubeAttentionPlaneSettings};
    use cubek_microkernel::attention::plane::{
        PlaneFragmentAttentionMicrokernelSettings, QueryHolder, ROW_STATE_HELD_BY_UNIT, ScoreHolder,
    };
    use cubek_microkernel::attention::{AttentionMicrokernel, D, G, Q, S};
    use cubek_microkernel::matmul::MatmulDim;
    use cubek_std::microkernel::{StageLayout, StageRing};

    type Settings = CubeStagedAttentionRoutineSettings;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
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
    /// for every head; one other than the default may ask for more than a cube holds, and says
    /// so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [
            offering(devices::m2_pro(), 8),
            offering(devices::h100(), 16),
        ] {
            for (position, settings) in CubeStagedAttentionRoutine::settings(&hw).iter().enumerate()
            {
                for problem in problems() {
                    let prepared = CubeStagedAttentionRoutine::prepare(&hw, &problem, settings);
                    let declined = matches!(
                        prepared,
                        Err(Decline::SharedMemoryPastDevice { .. } | Decline::CubePastDevice { .. })
                    );
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
        let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &f16, &Settings::DEFAULT)
            .expect("f16 prepares on an NVIDIA GPU's shapes");
        let CubeAttentionPlane::Fragment(plane) = blueprint.microkernel.plane else {
            panic!("the routine runs plane fragments")
        };
        let [(_, m), (_, n), (_, k)] = plane.score.instruction.shape();
        assert_eq!((m, n, k), (16, 16, 16));
    }

    /// On an NVIDIA GPU, which offers `f32` no instruction of its own, `f32` is read as tf32 and
    /// runs on its `16×16×8`: the plane's score is `16` keys by the whole head dim, which the tile
    /// walks in steps of `8`, and the value contracts its `16` keys in steps of `8`.
    #[test]
    fn f32_runs_on_tf32_where_the_device_offers_nothing_exact() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, _, f32] = problems();
        let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &f32, &Settings::DEFAULT)
            .expect("f32 prepares on tf32's 16×16×8");
        let CubeAttentionPlane::Fragment(plane) = blueprint.microkernel.plane else {
            panic!("the routine runs plane fragments")
        };
        assert_eq!(
            plane.read,
            AttentionRead::all(ElemType::Float(FloatKind::TF32))
        );
        assert_eq!(
            (
                plane.score.tile_extent(S),
                plane.score.tile_extent(D),
                plane.score.leaf_extent(MatmulDim::K)
            ),
            (16, 128, 8)
        );
        assert_eq!(
            (
                plane.value.tile_extent(S),
                plane.value.leaf_extent(MatmulDim::K)
            ),
            (16, 8)
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
        let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &f32, &through_registers)
            .expect("f32 prepares on tf32");
        assert_eq!(blueprint.microkernel.ring, StageRing::SINGLE);
    }

    /// A padded stage costs one chunk a row of its keys and values, and of the query staged beside
    /// them in the same layout, and a blocked one lays the same cells out at no cost: what the
    /// cube counts, so a layout that does not fit the device declines instead of overflowing.
    /// Padded blocks cost one chunk a row of a block whose row spans more than one: a stage block
    /// is a cache line wide along the keys' and values' dims, the query's an instruction's tile,
    /// whose row of eight `f16` is one chunk the tile DSL keeps in order.
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
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &problem, &settings)
                .expect("the lead's box prepares on the M2 in every layout");
            blueprint
        };
        let bytes = |layout| {
            let blueprint = prepared(layout);
            blueprint
                .microkernel
                .shared_memory(&hw, &item, &blueprint.shared_memory)
        };
        let microkernel = prepared(StageLayout::Strided).microkernel;
        let stage = item.stage(microkernel.seq_kv(), microkernel.key_order());
        let whole_rows: usize = stage
            .iter()
            .chain([&microkernel.staged_query(&item)])
            .map(|operand| operand.cells() / operand.space.extent(operand.innermost))
            .sum();
        let chunk = RowChunks::CHUNK_BYTES.max(hw.vector_ceiling(F16) * F16.size());
        let strided = bytes(StageLayout::Strided);
        assert_eq!(bytes(StageLayout::PaddedRows), strided + whole_rows * chunk);
        assert_eq!(bytes(StageLayout::Blocks), strided);
        let CubeAttentionPlane::Fragment(plane) = microkernel.plane else {
            panic!("the routine runs plane fragments")
        };
        assert_eq!(plane.out_edge() * F16.size(), RowChunks::CHUNK_BYTES);
        let block = plane.stage_tile();
        let padded_block_rows: usize = stage
            .iter()
            .map(|operand| {
                let (_, row) = *block
                    .iter()
                    .find(|(dimension, _)| *dimension == operand.innermost)
                    .expect("a stage block spans every dimension of the stage");
                let dim = operand.space.extent(operand.innermost);
                if operand.innermost != S {
                    assert_eq!(
                        row * F16.size(),
                        BLOCK_ROW_BYTES.min(dim * F16.size()),
                        "a block row spans a cache line, or its whole dim"
                    );
                }
                match row * F16.size() > RowChunks::CHUNK_BYTES {
                    true => operand.cells() / row,
                    false => 0,
                }
            })
            .sum();
        assert_eq!(
            bytes(StageLayout::PaddedBlocks),
            strided + padded_block_rows * chunk
        );
    }

    /// A device whose manual `mma` is `16 × 16`, 16 or 32 deep, in `f16` and in `e4m3`, and lays its
    /// accumulator as its `A`, and whose cubes hold 64 KiB of shared memory, as RDNA4's do.
    fn offering_square_mma() -> Hardware {
        let squares = [F16, E4M3].into_iter().flat_map(|element| {
            [16, 32].map(|k| MmaConfig {
                a_type: element,
                b_type: element,
                cd_type: F32,
                m: 16,
                n: 16,
                k,
            })
        });
        let mut hw = devices::offering_mma(offering(devices::h100(), 16), squares);
        devices::topology(&mut hw).max_shared_memory_size = 64 * 1024;
        hw
    }

    /// Planes holding their query in registers take its stage's bytes out of the cube's shared
    /// memory, so every head of a group of six shares each stage with four planes a head at a head
    /// dim of 256, where staging those planes' query does not fit.
    #[test]
    fn a_held_query_leaves_the_shared_memory_to_the_keys() {
        let hw = offering_square_mma();
        let problem = AttentionRoutineProblem::dense((1, 24, 4), (2048, 4096), (256, 256), F16);
        let item = problem.work_item();
        let shared = |query, heads, planes| {
            let lead = Settings::DEFAULT;
            let settings = Settings {
                microkernel: CubeAttentionMicrokernelSettings {
                    plane: CubeAttentionPlaneSettings::Fragment(
                        PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1).querying(query),
                    ),
                    heads: NonZeroUsize::new(heads).unwrap(),
                    planes_along_queries: NonZeroUsize::new(planes).unwrap(),
                    ..lead.microkernel
                },
                ..lead
            };
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &problem, &settings)?;
            Ok::<_, Decline>(blueprint.microkernel.shared_memory(
                &hw,
                &item,
                &blueprint.shared_memory,
            ))
        };
        let query = 16 * 256 * F16.size();
        assert_eq!(
            shared(QueryHolder::Stage, 1, 1).expect("one plane's query fits")
                - shared(QueryHolder::Registers, 1, 1).expect("a held query fits"),
            query
        );
        assert!(shared(QueryHolder::Registers, 6, 4).is_ok());
        assert!(matches!(
            shared(QueryHolder::Stage, 6, 4),
            Err(Decline::SharedMemoryPastDevice { .. })
        ));
    }

    /// An `e4m3` cache beside an `f16` query held in registers is multiplied on the FP8 instruction
    /// on both sides: the query is cast as it loads, the keys and values are staged as stored.
    #[test]
    fn a_held_query_reads_an_fp8_cache_as_it_is_stored() {
        let hw = offering_square_mma();
        let mut problem = AttentionRoutineProblem::dense((1, 24, 4), (2048, 4096), (256, 256), F16);
        problem.elements.key = E4M3;
        problem.elements.value = E4M3;
        let lead = Settings::DEFAULT;
        let settings = Settings {
            microkernel: CubeAttentionMicrokernelSettings {
                plane: CubeAttentionPlaneSettings::Fragment(
                    PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1)
                        .querying(QueryHolder::Registers),
                ),
                heads: NonZeroUsize::new(8).unwrap(),
                planes_along_queries: NonZeroUsize::new(4).unwrap(),
                ..lead.microkernel
            },
            ..lead
        };
        let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &problem, &settings)
            .expect("the held query reads the cache");
        let served = blueprint.microkernel.served(&problem.work_item()).elements;
        assert_eq!([served.query, served.key, served.value], [F16, E4M3, E4M3]);
    }

    /// A query held in registers is the `A` of a score in fragments: where the device gives no such
    /// score, as RDNA3's transposed accumulator, the setting is declined, not read some other way.
    #[test]
    fn a_held_query_asks_for_a_score_in_fragments() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, _, problem, _] = problems();
        let lead = Settings::DEFAULT;
        let held = Settings {
            microkernel: CubeAttentionMicrokernelSettings {
                plane: CubeAttentionPlaneSettings::Fragment(
                    PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1)
                        .querying(QueryHolder::Registers),
                ),
                ..lead.microkernel
            },
            ..lead
        };
        assert!(CubeStagedAttentionRoutine::prepare(&hw, &problem, &held).is_err());
    }

    /// A swizzled stage is one a cmma fragment cannot read, which takes its tiles off a pointer
    /// and a stride: the tile pads its rows instead, so the setting prepares the padded stage
    /// rather than declining.
    #[test]
    fn a_swizzled_stage_is_padded_under_cmma() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, _, problem, _] = problems();
        let storage = |layout| {
            let settings = Settings {
                microkernel: CubeAttentionMicrokernelSettings {
                    layout,
                    ..Settings::DEFAULT.microkernel
                },
                ..Settings::DEFAULT
            };
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &problem, &settings)
                .expect("the lead's box prepares on the M2");
            blueprint.shared_memory
        };
        assert_eq!(
            storage(StageLayout::SwizzledBlocks),
            storage(StageLayout::PaddedBlocks)
        );
    }

    /// A swizzled stage is listed only where the device offers the manual `mma`, the only
    /// transport that reads it swizzled: elsewhere it would lay the padded stage another setting
    /// already names, and a race would time that kernel twice.
    #[test]
    fn a_swizzled_stage_is_listed_only_beside_the_manual_mma() {
        let swizzled = |hw: &Hardware| {
            CubeStagedAttentionRoutine::settings(hw)
                .iter()
                .any(|settings| settings.microkernel.layout == StageLayout::SwizzledBlocks)
        };
        let manual = MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k: 16,
        };
        assert!(!swizzled(&offering(devices::m2_pro(), 8)));
        assert!(!swizzled(&offering(devices::h100(), 16)));
        assert!(swizzled(&devices::offering_mma(
            offering(devices::h100(), 16),
            [manual]
        )));
    }

    /// The copy engine fills a stage where the device has one and the ring lands the lines in
    /// shared memory, and only lines of 4, 8 or 16 bytes, the copies it makes. A ring that
    /// prefetches through the units' registers, or a device with no engine, has the units copy.
    #[test]
    fn the_copy_engine_fills_the_stages_the_ring_lands_in_shared_memory() {
        let hw = offering_nvidia(devices::h100());
        let [_, _, problem, _] = problems();
        let item = problem.work_item();
        let prepared = |hw: &Hardware, ring| {
            let settings = Settings {
                microkernel: CubeAttentionMicrokernelSettings {
                    ring,
                    delivery: CubeStagedDelivery::AsyncCopy,
                    ..Settings::DEFAULT.microkernel
                },
                ..Settings::DEFAULT
            };
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(hw, &problem, &settings)
                .expect("the lead prepares on any ring");
            blueprint
        };
        let blueprint = prepared(&hw, StageRing::DOUBLE);
        assert_eq!(
            blueprint.microkernel.delivery,
            CubeStagedDelivery::AsyncCopy
        );
        assert_eq!(
            blueprint.microkernel.delivery(&blueprint.widths, &item),
            Delivery::AsyncPerUnit
        );
        let single_cells = blueprint.widths.map(|_, _| 1);
        assert_eq!(
            blueprint.microkernel.delivery(&single_cells, &item),
            Delivery::SyncPerUnit
        );
        let through_registers = prepared(&hw, StageRing::DOUBLE_THROUGH_REGISTERS);
        assert_eq!(
            through_registers.microkernel.delivery,
            CubeStagedDelivery::Copy
        );
        let mut no_engine = hw.clone();
        devices::features(&mut no_engine).copy_async = false;
        assert_eq!(
            prepared(&no_engine, StageRing::DOUBLE).microkernel.delivery,
            CubeStagedDelivery::Copy
        );
    }

    /// A cube's box is its heads over its queries, a plane each, and its stage is that box against
    /// a stage of keys: the keys are staged once for every head the cube holds.
    #[test]
    fn a_cube_stages_its_keys_once_for_every_head() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, problem, _, _] = problems();
        let (blueprint, grid) =
            CubeStagedAttentionRoutine::prepare(&hw, &problem, &Settings::DEFAULT)
                .expect("the default prepares on the M2");
        let (cube, stage) = (
            blueprint.partitioning.cube_space(),
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
        let holder = |settings: CubeAttentionMicrokernelSettings| {
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(
                &hw,
                &problems()[2],
                &Settings {
                    microkernel: settings,
                    ..Settings::DEFAULT
                },
            )
            .expect("prepares where both cmma and the manual mma are offered");
            let CubeAttentionPlane::Fragment(plane) = blueprint.microkernel.plane else {
                panic!("the routine runs plane fragments")
            };
            plane.score_holder()
        };
        let settings = CubeStagedAttentionRoutine::settings(&hw);
        let lead = settings[0].microkernel;
        assert_eq!(holder(lead), ScoreHolder::Fragments);
        for split in settings
            .iter()
            .filter(|settings| settings.microkernel.planes_along_keys.get() > 1)
        {
            assert_eq!(holder(split.microkernel), ScoreHolder::Window, "{split}");
        }
    }

    /// Over a packed cache the kernel opens the output's sum on values its stages decode, which
    /// fragments held as the value's `A` name no element to contract against: where the lead
    /// keeps its score in fragments over a float cache, it reads it through its windows over a
    /// packed one, as the kernel can expand.
    #[test]
    fn a_packed_cache_is_read_with_its_score_through_the_windows() {
        use crate::kernels::{AttentionCacheLayout, KeyFixup, KvBitWidths, PackedLayout};
        let manual = |k| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k,
        };
        let hw = devices::offering_mma(offering(devices::h100(), 16), [manual(16), manual(8)]);
        let holder = |problem: &AttentionRoutineProblem| {
            let (blueprint, _) =
                CubeStagedAttentionRoutine::prepare(&hw, problem, &Settings::DEFAULT)
                    .expect("the lead prepares where both cmma and the manual mma are offered");
            let CubeAttentionPlane::Fragment(plane) = blueprint.microkernel.plane else {
                panic!("the routine runs plane fragments")
            };
            plane.score_holder()
        };
        let [_, _, float, _] = problems();
        let packed = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Packed(PackedLayout {
                bits: KvBitWidths { key: 4, value: 4 },
                fixup: KeyFixup::RoundOnly,
            }),
            ..float.clone()
        };
        assert_eq!(holder(&float), ScoreHolder::Fragments);
        assert_eq!(holder(&packed), ScoreHolder::Window);
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
        let prepared = CubeStagedAttentionRoutine::prepare(
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

    /// The default takes the boxes of queries in order, and every listed setting after it takes
    /// them last first, the lead holding its score the whole head dim deep among them: the lead
    /// is listed both ways, so a race still weighs the order.
    #[test]
    fn the_listed_settings_take_the_queries_last_first() {
        let hw = offering(devices::m2_pro(), 8);
        let [_, _, problem, _] = problems();
        let reverses_queries = |settings: &Settings| {
            let (blueprint, _) = CubeStagedAttentionRoutine::prepare(&hw, &problem, settings)
                .expect("the lead's box prepares on the M2 in either order");
            blueprint
                .partitioning
                .as_tile()
                .levels()
                .iter()
                .any(|level| level.is_reversed(Q))
        };
        let settings = CubeStagedAttentionRoutine::settings(&hw);
        assert_eq!(settings[0], Settings::DEFAULT);
        assert!(
            settings[1..]
                .iter()
                .all(|settings| settings.queries == QueryOrder::LastFirst)
        );
        let reversed = Settings {
            queries: QueryOrder::LastFirst,
            ..Settings::DEFAULT
        };
        assert_eq!(settings[1], reversed);
        assert!(!reverses_queries(&settings[0]));
        assert!(reverses_queries(&settings[1]));
    }
}
