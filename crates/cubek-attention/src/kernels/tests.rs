//! Tests for the attention family level: the seam it shares, and the
//! cross-kernel selector over it.

use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{AttentionRoutineProblem, AttentionVisibility, ScoreScale};
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubek_std::Hardware;
use cubek_std::devices;

/// `hw` offering the square `edge`-wide instruction for `f16`, `bf16` and `f32` operands summed in
/// `f32`.
fn offering(hw: Hardware, edge: u32) -> Hardware {
    let square = |element| MmaConfig {
        a_type: element,
        b_type: element,
        cd_type: ElemType::Float(FloatKind::F32),
        m: edge,
        n: edge,
        k: edge,
    };
    devices::offering_cmma(
        hw,
        [
            square(ElemType::Float(FloatKind::F16)),
            square(ElemType::Float(FloatKind::BF16)),
            square(ElemType::Float(FloatKind::F32)),
        ],
    )
}

/// Qwen3-8B's attention at `f16`, 32 heads over 8 of 128: `queries` positions over `keys`, causal
/// past one position.
fn qwen(queries: usize, keys: usize) -> AttentionRoutineProblem {
    let f16 = ElemType::Float(FloatKind::F16);
    let visibility = AttentionVisibility::new(queries > 1, None);
    let key = AttentionKey::new(
        f16,
        32,
        8,
        128,
        queries,
        keys,
        1,
        visibility,
        ScoreScale::standard(128),
        None,
    );
    AttentionRoutineProblem::try_from(&key).expect("an even group")
}

mod base {
    use super::super::select::*;

    /// Everything that is not a pin lets the caller's policy choose; a `Specific`
    /// names the kernel a benchmark row or a capture asked for and must not be
    /// substituted.
    #[test]
    fn the_policies_are_the_strategies_that_name_no_kernel() {
        let policies: Vec<_> = AttentionOpStrategy::all()
            .into_iter()
            .filter(AttentionOpStrategy::is_policy)
            .collect();
        assert_eq!(
            policies,
            vec![
                AttentionOpStrategy::Autotune,
                AttentionOpStrategy::Heuristic(AttentionKernelHeuristic::Tile),
                AttentionOpStrategy::Heuristic(AttentionKernelHeuristic::Cubek),
            ],
        );
    }

    /// Autotune is what a caller naming nothing gets, as in the matmul family.
    #[test]
    fn the_default_is_autotune() {
        assert_eq!(
            AttentionOpStrategy::default(),
            AttentionOpStrategy::Autotune
        );
    }
}

mod selector {
    use super::super::select::selector::*;
    use super::super::*;
    use AttentionChoice::{Autotune as Measured, Kernel};
    use AttentionKernelHeuristic as H;
    use AttentionOpStrategy::{Autotune, Heuristic, Specific};
    use AttentionSelection::{DequantizeThen, Float, Packed};
    use KernelChoice::{Pin, Rule};
    use QueryPositions::{Many, One};

    const KINDS: [KvKind; 2] = [KvKind::Float, KvKind::Packed];
    const LAST_ARRIVAL_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);
    const SECOND_LAUNCH_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::SecondLaunchShuffle);
    const CUBE_FRAGMENT: AttentionKernel = AttentionKernel::Cube(CubeAttentionKernel::Fragment);

    /// What a packed cache of one query position reads under every policy: the hypercube shuffle
    /// merged on the last arrival by rule, then the unit routine over the rebuilt cache.
    const PACKED_BY_RULE: AttentionSelection = Packed {
        read: Rule(LAST_ARRIVAL_SHUFFLE),
        otherwise: Some(Kernel(Rule(AttentionKernel::Unit))),
    };

    /// The tile ceiling: over a float cache the hypercube by rule at one query position and the
    /// cube elsewhere; over a packed one the hypercube decoding inline at one position, the unit
    /// routine after it, and the cache rebuilt before the cube elsewhere. Nothing measured
    /// anywhere.
    #[test]
    fn the_tile_ceiling_rules() {
        let tile = Heuristic(H::Tile);
        assert_eq!(
            select(tile, One, KvKind::Float),
            Ok(Float(Kernel(Rule(LAST_ARRIVAL_SHUFFLE))))
        );
        assert_eq!(
            select(tile, Many, KvKind::Float),
            Ok(Float(Kernel(Rule(CUBE_FRAGMENT)))),
            "many query positions are never measured below the burn ceiling"
        );
        assert_eq!(select(tile, One, KvKind::Packed), Ok(PACKED_BY_RULE));
        assert_eq!(
            select(tile, Many, KvKind::Packed),
            Ok(DequantizeThen(Kernel(Rule(CUBE_FRAGMENT))))
        );
    }

    /// Where a kernel a rule named declines, the cube's plane unit groups run after its plane
    /// fragments, the hypercube shuffle merged by a second launch after the one merged on the last
    /// arrival, and the unit routine after any other: every kernel reaches the unit routine,
    /// and nothing follows it.
    #[test]
    fn every_rule_ends_at_the_unit_routine() {
        assert_eq!(
            CUBE_FRAGMENT.instead_by_rule(),
            Some(AttentionKernel::Cube(CubeAttentionKernel::UnitGroup))
        );
        assert_eq!(
            LAST_ARRIVAL_SHUFFLE.instead_by_rule(),
            Some(SECOND_LAUNCH_SHUFFLE)
        );
        for kernel in AttentionKernel::all() {
            let last = std::iter::successors(Some(kernel), AttentionKernel::instead_by_rule).last();
            assert_eq!(last, Some(AttentionKernel::Unit), "{kernel:?}");
        }
        assert_eq!(AttentionKernel::Unit.instead_by_rule(), None);
    }

    /// `Autotune` measures every float call, on the table of its query positions; a packed call
    /// of one position has one kernel and nothing to race, and a packed call of many is rebuilt
    /// and measured as a float one.
    #[test]
    fn autotune_measures_every_float_call() {
        for queries in [One, Many] {
            assert_eq!(
                select(Autotune, queries, KvKind::Float),
                Ok(Float(Measured(queries)))
            );
        }
        assert_eq!(select(Autotune, One, KvKind::Packed), Ok(PACKED_BY_RULE));
        assert_eq!(
            select(Autotune, Many, KvKind::Packed),
            Ok(DequantizeThen(Measured(Many)))
        );
    }

    /// Every candidate of the family is on a table, so the cubek ceiling and `Autotune` name the
    /// same set.
    #[test]
    fn the_cubek_ceiling_and_autotune_agree_while_every_candidate_is_on_a_table() {
        for cache in KINDS {
            for queries in [One, Many] {
                assert_eq!(
                    select(Heuristic(H::Cubek), queries, cache),
                    select(Autotune, queries, cache),
                    "{queries:?} over {cache:?}"
                );
            }
        }
    }

    /// A routine's pin runs that routine at any query length over a float cache; over a packed
    /// one only the two hypercube shuffles' run.
    #[test]
    fn a_routine_pin_runs_its_routine() {
        for kernel in AttentionKernel::all() {
            for queries in [One, Many] {
                assert_eq!(
                    select(Specific(kernel), queries, KvKind::Float),
                    Ok(Float(Kernel(Pin(kernel))))
                );
            }
            let packed = select(Specific(kernel), One, KvKind::Packed);
            match kernel {
                LAST_ARRIVAL_SHUFFLE | SECOND_LAUNCH_SHUFFLE => assert_eq!(
                    packed,
                    Ok(Packed {
                        read: Pin(kernel),
                        otherwise: None
                    })
                ),
                _ => assert!(packed.is_err(), "{kernel:?}"),
            }
        }
    }
}

mod problem {
    use super::super::*;
    use super::{offering, qwen};
    use cubek_microkernel::attention::plane::{
        PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
    };
    use cubek_microkernel::attention::{AttentionMicrokernel, S};
    use cubek_std::Hardware;
    use cubek_std::devices;

    /// Every power of two up to the widest class, the last inside the limit standing for it.
    #[test]
    fn capacity_classes_are_the_powers_of_two_to_the_widest() {
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(1 << 20),
            vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        );
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(100),
            vec![2, 4, 8, 16, 32, 64, 100]
        );
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(64),
            vec![2, 4, 8, 16, 32, 64]
        );
    }

    /// The widest step is a multiple of every step of keys a routine walks, at every setting it
    /// lists on every device the fleet names and on an M2 Pro offering its square instruction, so
    /// a cache of whole widest steps is read unmasked; and the widest class is a multiple of it,
    /// so the classes compile every setting's launch over whole steps as well as its bounded one.
    #[test]
    fn every_step_of_keys_divides_the_widest_class() {
        assert!(
            AttentionRoutineProblem::WIDEST_CAPACITY_CLASS
                .is_multiple_of(AttentionRoutineProblem::WIDEST_STEP)
        );
        let fleet = [
            offering(devices::m2_pro(), 8),
            devices::cpu_12(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::m2_pro(),
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::h100(),
        ];
        let prepared = [
            fleet
                .iter()
                .map(|hw| assert_steps_divide(UnitAttentionRoutine, hw))
                .sum::<usize>(),
            fleet
                .iter()
                .map(|hw| assert_steps_divide(PlaneFragmentAttentionRoutine, hw))
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(
                        CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
                        hw,
                    )
                })
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw)
                })
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(HypercubeSecondLaunchShuffleAttentionRoutine::new(), hw)
                })
                .sum(),
        ];
        assert!(
            prepared.iter().all(|&count| count > 0),
            "prepared {prepared:?}"
        );
    }

    /// A problem read from pools of pages reads its operands as stored: a table binds no
    /// operand as words, so an `f32` cache is never read as tf32.
    #[test]
    fn a_paged_problem_reads_as_stored() {
        const F32: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32);
        let f32 = AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F32);
        let paged = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size: 16 },
            ..f32
        };
        assert_eq!(f32.work_item().readings().count(), 2);
        assert_eq!(paged.work_item().readings().collect::<Vec<_>>(), [F32]);
    }

    /// A cube whose planes read keys where they lie walks a paged cache a page at a time only
    /// where a page holds a whole number of its steps, more than one; a staging cube never does.
    #[test]
    fn a_cube_walks_pages_only_where_a_page_holds_several_whole_steps() {
        use cubek_std::routine::Routine;
        const F16: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16);
        let hw = devices::m2_pro();
        let dense = AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F16);
        let paged = |page_size| AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size },
            ..dense
        };
        let shuffle = &CubeShuffleAttentionRoutine.settings(&hw)[0];
        let page_steps = |problem: &AttentionRoutineProblem| {
            let (blueprint, _) = CubeShuffleAttentionRoutine
                .prepare(&hw, problem, shuffle)
                .expect("the decode step prepares");
            blueprint
                .microkernel
                .page_steps
                .map(core::num::NonZeroUsize::get)
        };
        let keys = CubeShuffleAttentionRoutine
            .prepare(&hw, &dense, shuffle)
            .expect("the decode step prepares")
            .0
            .microkernel
            .keys();
        assert_eq!(page_steps(&dense), None);
        assert_eq!(page_steps(&paged(4 * keys)), Some(4));
        assert_eq!(page_steps(&paged(keys)), None);
        assert_eq!(page_steps(&paged(3 * keys / 2)), None);
        let staged = CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new();
        let staged_settings = &staged.settings(&hw)[0];
        let (blueprint, _) = staged
            .prepare(&hw, &paged(4 * keys), staged_settings)
            .expect("the decode step prepares");
        assert_eq!(blueprint.microkernel.page_steps, None);
    }

    /// Every routine but the two hypercube shuffles declines a packed cache, at every setting it
    /// lists; each hypercube shuffle prepares it.
    #[test]
    fn only_the_hypercube_shuffles_read_a_packed_cache() {
        fn reads<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
            routine: R,
            hw: &Hardware,
            problem: &AttentionRoutineProblem,
        ) -> bool {
            routine
                .settings(hw)
                .iter()
                .any(|settings| routine.prepare(hw, problem, settings).is_ok())
        }
        fn declines<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
            routine: R,
            hw: &Hardware,
            problem: &AttentionRoutineProblem,
        ) {
            for settings in routine.settings(hw) {
                assert_eq!(
                    routine.prepare(hw, problem, &settings).err(),
                    Some(ATTENTION_PACKED_UNREAD),
                    "{settings}"
                );
            }
        }
        let hw = offering(devices::h100(), 16);
        let layout = PackedLayout {
            bits: cubek_microkernel::attention::packed_cache::KvBitWidths { key: 4, value: 4 },
            fixup: cubek_microkernel::attention::packed_cache::KeyFixup::RoundOnly,
        };
        let packed = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Packed(layout),
            ..qwen(1, 4096)
        };
        assert!(reads(
            HypercubeLastArrivalShuffleAttentionRoutine::new(),
            &hw,
            &packed
        ));
        assert!(reads(
            HypercubeSecondLaunchShuffleAttentionRoutine::new(),
            &hw,
            &packed
        ));
        declines(UnitAttentionRoutine, &hw, &packed);
        declines(PlaneFragmentAttentionRoutine, &hw, &packed);
        declines(
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &hw,
            &packed,
        );
        declines(
            CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new(),
            &hw,
            &packed,
        );
        declines(CubeShuffleAttentionRoutine, &hw, &packed);
        declines(HypercubeSecondLaunchFragmentAttentionRoutine, &hw, &packed);
    }

    fn assert_steps_divide<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
        routine: R,
        hw: &Hardware,
    ) -> usize {
        let problems = [qwen(512, 512), qwen(1, 4096)];
        let mut prepared = 0;
        for settings in routine.settings(hw) {
            for problem in &problems {
                let Ok((blueprint, _)) = routine.prepare(hw, problem, &settings) else {
                    continue;
                };
                prepared += 1;
                let step = blueprint.partitioning.stage().extent(S);
                assert!(
                    AttentionRoutineProblem::WIDEST_STEP.is_multiple_of(step),
                    "{settings} on {hw:?} steps {step} keys"
                );
            }
        }
        prepared
    }

    /// A single position reads its group as rows of one head, every one attending every key: a
    /// causal single position sees every attended key, so nothing is masked differently. More
    /// positions keep their own rows and visibility.
    #[test]
    fn a_single_position_reads_its_group_as_rows() {
        assert_eq!(
            QueryAxis::new(1).read(4, 1, AttentionVisibility::Causal),
            (1, 4, AttentionVisibility::Attended)
        );
        assert_eq!(
            QueryAxis::new(7).read(4, 7, AttentionVisibility::Causal),
            (4, 7, AttentionVisibility::Causal)
        );
    }

    /// A windowed single position keeps its window as rows: every head of the group sees the
    /// last keys the window spans, which no longer depends on which row it is. More positions
    /// keep their own window.
    #[test]
    fn a_single_position_keeps_its_window_as_rows() {
        let window = core::num::NonZeroUsize::new(512).unwrap();
        let windowed = AttentionVisibility::new(true, Some(window));
        assert_eq!(windowed, AttentionVisibility::Windowed { window });
        assert_eq!(
            QueryAxis::new(1).read(4, 1, windowed),
            (1, 4, AttentionVisibility::Recent { window })
        );
        assert_eq!(QueryAxis::new(7).read(4, 7, windowed), (4, 7, windowed));
        assert!(windowed.is_causal());
        assert!(!AttentionVisibility::Recent { window }.is_causal());
    }
}

mod launch {
    use super::super::tune::key::AttentionKey;
    use super::super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::AttentionMicrokernel;
    use cubek_microkernel::attention::plane::PlaneFragmentAttentionMicrokernel;
    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::microkernel::KernelEstimate;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Every routine's estimate at every setting it lists, on a prompt and on a decode step: each
    /// owner down the walk set its own part, and the cubes perform at least what the problem asks.
    #[test]
    fn every_routine_estimates_every_setting_whole() {
        assert_every_estimate_is_whole(UnitAttentionRoutine);
        assert_every_estimate_is_whole(PlaneFragmentAttentionRoutine);
        assert_every_estimate_is_whole(CubeStagedAttentionRoutine::<
            PlaneFragmentAttentionMicrokernel,
        >::new());
        assert_every_estimate_is_whole(HypercubeLastArrivalShuffleAttentionRoutine::new());
        assert_every_estimate_is_whole(HypercubeSecondLaunchShuffleAttentionRoutine::new());
    }

    /// An M2 Pro offering the square `8×8×8` instruction for `f16` and `f32` summed in `f32`.
    fn m2_pro() -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: 8,
            n: 8,
            k: 8,
        };
        devices::offering_cmma(devices::m2_pro(), [square(F16), square(F32)])
    }

    /// Qwen3-8B's attention: a prompt of 512 positions, and a decode step against a cache of 4096,
    /// the problems the tables read off their keys.
    fn problems() -> [AttentionRoutineProblem; 2] {
        [(512, 512, true), (1, 4096, false)].map(|(queries, keys, causal)| {
            let visibility = AttentionVisibility::new(causal, None);
            let key = AttentionKey::new(
                F16,
                32,
                8,
                128,
                queries,
                keys,
                1,
                visibility,
                ScoreScale::standard(128),
                None,
            );
            AttentionRoutineProblem::try_from(&key).expect("an even group")
        })
    }

    fn assert_every_estimate_is_whole<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
        routine: R,
    ) {
        let hw = &m2_pro();
        let mut prepared = 0;
        for settings in routine.settings(hw) {
            for problem in problems() {
                let Ok((blueprint, grid)) = routine.prepare(hw, &problem, &settings) else {
                    continue;
                };
                prepared += 1;
                let mut estimate = KernelEstimate::from(&grid);
                routine.estimate(&problem, &blueprint, &grid, &mut estimate);
                let (staging, cubes) = (estimate.staging.as_ref(), estimate.cubes);
                assert!(estimate.work.is_some(), "{settings}: no work");
                let staging = staging.unwrap_or_else(|| panic!("{settings}: no staging"));
                assert!(
                    staging.stages > 0 && !staging.reads.is_empty(),
                    "{settings}"
                );
                let performed = staging.operations * cubes.expect("a static grid") as f64;
                let useful = problem.demand().compute_ops as f64;
                assert!(performed >= useful, "{settings}: {performed} < {useful}");
            }
        }
        assert!(prepared > 0, "no setting of the routine prepares here");
    }
}

/// The routines' candidate lists reach every value a table needs on each lever: the step of
/// keys, the box of queries, the cuts of the keys across cubes and planes.
mod routines {
    use super::super::*;
    use super::{offering, qwen};
    use cubecl::tile::stage::Prefetch;
    use cubek_microkernel::attention::cube::CubeAttentionMicrokernelSettings;
    use cubek_microkernel::attention::plane::{
        PlaneFragmentAttentionMicrokernel, SWIZZLED_STAGE_UNREAD,
    };
    use cubek_microkernel::attention::{AttentionMicrokernel, CubeAttentionStageLayout, Q, S};
    use cubek_microkernel::staging::StageRing;
    use cubek_std::Hardware;
    use cubek_std::Instructions;
    use cubek_std::devices;
    use cubek_std::routine::Routine;
    use std::collections::BTreeSet;

    /// The keys a step covers and the query positions a cube's box holds, at every setting
    /// `routine` lists that prepares for `problem` on `hw`.
    fn reach<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
        routine: R,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
    ) -> (BTreeSet<usize>, BTreeSet<usize>) {
        let (mut steps, mut queries) = (BTreeSet::new(), BTreeSet::new());
        for settings in routine.settings(hw) {
            if let Ok((blueprint, _)) = routine.prepare(hw, problem, &settings) {
                steps.insert(blueprint.partitioning.stage().extent(S));
                queries.insert(blueprint.partitioning.cube_box().extent(Q));
            }
        }
        (steps, queries)
    }

    /// A prompt's step reaches 256 keys on a device whose shared memory holds it, and its box
    /// goes down to a few positions: the widest and narrowest a prompt table needs to race.
    #[test]
    fn a_prompt_reaches_wide_steps_and_narrow_boxes() {
        let hw = offering(devices::h100(), 16);
        let prompt = qwen(512, 512);
        let (cube_steps, _) = reach(
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &hw,
            &prompt,
        );
        assert!(cube_steps.contains(&256), "cube steps {cube_steps:?}");
        let (_, few) = reach(
            HypercubeLastArrivalShuffleAttentionRoutine::new(),
            &hw,
            &qwen(4, 512),
        );
        assert!(
            few.iter().any(|&queries| queries <= 4),
            "hypercube boxes {few:?}"
        );
    }

    /// A decode step's cache is cut across two to thirty-two cubes, each cube's planes into up to
    /// eight runs of it, a cube's planes alone into up to eight, and its step spans one key a team
    /// to thirty-two a plane.
    #[test]
    fn a_decode_step_reaches_every_split_and_step() {
        let hw = offering(devices::m2_pro(), 8);
        let settings = HypercubeLastArrivalShuffleAttentionRoutine::new().settings(&hw);
        let cubes: BTreeSet<usize> = settings.iter().map(|s| s.cubes_along_keys.get()).collect();
        for count in [2, 4, 8, 16, 32] {
            assert!(cubes.contains(&count), "cubes along keys {cubes:?}");
        }
        let cube_planes: BTreeSet<usize> = settings
            .iter()
            .map(|s| s.microkernel.cube.planes_along_keys.get())
            .collect();
        for count in [1, 4, 8] {
            assert!(
                cube_planes.contains(&count),
                "planes along keys {cube_planes:?}"
            );
        }
        let planes: BTreeSet<usize> = CubeShuffleAttentionRoutine
            .settings(&hw)
            .iter()
            .map(|s| s.microkernel.planes_along_keys.get())
            .collect();
        assert!(planes.contains(&8), "planes along keys {planes:?}");
        let (steps, _) = reach(
            HypercubeLastArrivalShuffleAttentionRoutine::new(),
            &hw,
            &qwen(1, 4096),
        );
        assert!(
            steps.first() <= Some(&2) && steps.last() >= Some(&32),
            "hypercube steps {steps:?}"
        );
    }

    /// The cube's table reaches every other ring of stages and every other layout of a stage, each
    /// changed alone from the lead, so a race can tell whether loads in flight or a stage spread
    /// over the banks pay on a device; and a prompt's stage is small enough that a ring holding
    /// its next stage in registers keeps it there.
    #[test]
    fn the_cube_reaches_every_ring_and_layout() {
        let hw = offering(devices::m2_pro(), 8);
        let settings =
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new().settings(&hw);
        let lead = CubeStagedAttentionRoutineSettings::<PlaneFragmentAttentionMicrokernel>::DEFAULT
            .microkernel;
        for ring in [
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
            StageRing::DOUBLE_THROUGH_REGISTERS,
        ] {
            let listed = CubeAttentionMicrokernelSettings { ring, ..lead };
            assert!(
                settings.iter().any(|s| s.microkernel == listed),
                "{listed} is not listed"
            );
        }
        for layout in [
            CubeAttentionStageLayout::PaddedRows,
            CubeAttentionStageLayout::Blocks,
            CubeAttentionStageLayout::PaddedBlocks,
        ] {
            let listed = CubeAttentionMicrokernelSettings { layout, ..lead };
            assert!(
                settings.iter().any(|s| s.microkernel == listed),
                "{listed} is not listed"
            );
        }
        let prefetching = CubeStagedAttentionRoutineSettings::<PlaneFragmentAttentionMicrokernel> {
            microkernel: CubeAttentionMicrokernelSettings {
                ring: StageRing::SINGLE_THROUGH_REGISTERS,
                ..lead
            },
            ..CubeStagedAttentionRoutineSettings::<PlaneFragmentAttentionMicrokernel>::DEFAULT
        };
        let (blueprint, _) = CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new()
            .prepare(&hw, &qwen(512, 512), &prefetching)
            .expect("the lead's box prepares on the M2");
        assert_eq!(blueprint.microkernel.ring.prefetch, Prefetch::InRegisters);
    }

    /// Every setting the hypercube fragment routine lists prepares, a decode step and a prompt's
    /// chunk alike, where the device offers a square instruction: its cubes of one plane along
    /// the keys, merged by a second launch, fit the shared memory of the devices it is raced on.
    /// Past the lead, a setting whose stages are swizzled declines on a device whose fragments
    /// read the score through windows alone, and on no other grounds.
    #[test]
    fn every_hypercube_fragment_setting_prepares() {
        for hw in [
            offering(devices::m2_pro(), 8),
            offering(devices::h100(), 16),
        ] {
            let listed = HypercubeSecondLaunchFragmentAttentionRoutine.settings(&hw);
            for (position, settings) in listed.iter().enumerate() {
                for problem in [qwen(1, 4096), qwen(16, 1000)] {
                    match HypercubeSecondLaunchFragmentAttentionRoutine
                        .prepare(&hw, &problem, settings)
                    {
                        Ok(_) => {}
                        Err(why) if position > 0 && why == SWIZZLED_STAGE_UNREAD => {}
                        prepared => panic!("{settings} on {hw:?}: {prepared:?}"),
                    }
                }
            }
        }
    }

    /// `bf16` on Metal has no square instruction for the cube's plane fragments; on an H100 every
    /// float element does.
    #[test]
    fn a_square_instruction_serves_every_float_on_an_h100_and_not_on_metal() {
        assert!(
            !devices::m2_pro()
                .properties
                .features
                .matmul
                .cmma
                .iter()
                .copied()
                .serves_attention_in_every_float()
        );
        assert!(
            offering(devices::h100(), 16)
                .properties
                .features
                .matmul
                .cmma
                .iter()
                .copied()
                .serves_attention_in_every_float()
        );
    }
}
