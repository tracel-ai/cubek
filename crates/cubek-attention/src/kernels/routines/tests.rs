//! The routines' candidate lists reach every value autotune needs on each lever: the step of
//! keys, the box of queries, the cuts of the keys across cubes and planes.

use crate::kernels::test_support::{offering, qwen};
use crate::kernels::*;
use cubecl::tile::stage::Prefetch;
use cubek_microkernel::attention::cube::CubeAttentionMicrokernelSettings;
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionOperands, Q, S};
use cubek_std::microkernel::{SWIZZLED_STAGE_UNREAD, StageLayout, StageRing};
use cubek_std::routine::Routine;
use cubek_std::{Hardware, devices};
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
            queries.insert(blueprint.partitioning.cube_space().extent(Q));
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
    let (cube_steps, _) = reach(CubeStagedAttentionRoutine, &hw, &prompt);
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
    let settings = CubeStagedAttentionRoutine.settings(&hw);
    let lead = CubeStagedAttentionRoutineSettings::DEFAULT.microkernel;
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
        StageLayout::PaddedRows,
        StageLayout::Blocks,
        StageLayout::PaddedBlocks,
    ] {
        let listed = CubeAttentionMicrokernelSettings { layout, ..lead };
        assert!(
            settings.iter().any(|s| s.microkernel == listed),
            "{listed} is not listed"
        );
    }
    let prefetching = CubeStagedAttentionRoutineSettings {
        microkernel: CubeAttentionMicrokernelSettings {
            ring: StageRing::SINGLE_THROUGH_REGISTERS,
            ..lead
        },
        ..CubeStagedAttentionRoutineSettings::DEFAULT
    };
    let (blueprint, _) = CubeStagedAttentionRoutine
        .prepare(&hw, &qwen(512, 512), &prefetching)
        .expect("the lead's box prepares on the M2");
    assert_eq!(blueprint.microkernel.ring.prefetch, Prefetch::InRegisters);
}

/// Every setting the hypercube fragment routine lists prepares, a decode step and a prompt's
/// chunk alike, where the device offers a square instruction: its cubes of one plane along
/// the keys fit the shared memory of the devices it is raced on.
/// Past the lead, a setting whose stages are swizzled declines on a device whose fragments
/// read the score through windows alone, and on no other grounds.
#[test]
fn every_hypercube_fragment_setting_prepares() {
    for hw in [
        offering(devices::m2_pro(), 8),
        offering(devices::h100(), 16),
    ] {
        let listed = HypercubeFragmentPartialAttentionRoutine.settings(&hw);
        for (position, settings) in listed.iter().enumerate() {
            for problem in [qwen(1, 4096), qwen(16, 1000)] {
                match HypercubeFragmentPartialAttentionRoutine.prepare(&hw, &problem, settings) {
                    Ok(_) => {}
                    Err(why) if position > 0 && why == SWIZZLED_STAGE_UNREAD => {}
                    prepared => panic!("{settings} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }
}

const E4M3: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::E4M3);
const F16: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16);

/// A problem of `f16` queries over a cache whose keys are stored as `key` and values as `value`.
fn cached(
    queries: usize,
    keys: usize,
    key: cubecl::ir::ElemType,
    value: cubecl::ir::ElemType,
) -> AttentionRoutineProblem {
    let f16 = qwen(queries, keys);
    AttentionRoutineProblem {
        elements: AttentionOperands {
            key,
            value,
            ..f16.elements
        },
        ..f16
    }
}

/// The caches an FP8 setting stores: `e4m3` keys, `e4m3` values, and both.
const FP8_CACHES: [(cubecl::ir::ElemType, cubecl::ir::ElemType); 3] =
    [(E4M3, F16), (F16, E4M3), (E4M3, E4M3)];

/// An RTX PRO 6000 offering the manual `mma`'s 16-bit forms and its `e4m3` one.
fn offering_fp8() -> Hardware {
    let (f16, f32, e4m3) = (
        cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16),
        cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::E4M3),
    );
    let form = |a, k| cubecl::features::MmaConfig {
        a_type: a,
        b_type: a,
        cd_type: f32,
        m: 16,
        n: 8,
        k,
    };
    devices::offering_mma(
        devices::rtx_pro_6000(),
        [form(f16, 8), form(f16, 16), form(e4m3, 32)],
    )
}

/// Whether any setting `routine` lists prepares for `problem` on `hw`.
fn prepares<R: Routine<Problem = AttentionRoutineProblem>>(
    routine: R,
    hw: &Hardware,
    problem: &AttentionRoutineProblem,
) -> bool {
    routine
        .settings(hw)
        .iter()
        .any(|settings| routine.prepare(hw, problem, settings).is_ok())
}

/// The reads the fragment settings that prepare for `problem` on `hw` take.
fn fragment_reads(hw: &Hardware, problem: &AttentionRoutineProblem) -> BTreeSet<(String, String)> {
    let fragments = HypercubeFragmentPartialAttentionRoutine;
    fragments
        .settings(hw)
        .iter()
        .filter_map(|settings| fragments.prepare(hw, problem, settings).ok())
        .map(|(blueprint, _)| {
            let served = blueprint.microkernel.served(&problem.work_item()).elements;
            (format!("{:?}", served.key), format!("{:?}", served.value))
        })
        .collect()
}

/// Every FP8 cache is read at a prompt and at a decode step. The plane fragments multiply `e4m3`
/// keys and `e4m3` values on the device's FP8 instruction, the `f16` query converted ahead and the
/// probabilities cast to `e4m3` beside the values; on a device without the instruction each side
/// is widened to the query's `f16` by its stage as it fills. The plane shuffles widen every
/// operand into the sum as they read it. The unit routine, which computes in its operands' one
/// element, declines them all.
#[test]
fn every_fp8_cache_is_read_at_a_prompt_and_at_a_decode_step() {
    let hw = offering_fp8();
    let without = offering(devices::rtx_pro_6000(), 16);
    let (e4m3, f16) = (format!("{E4M3:?}"), format!("{F16:?}"));
    for (key, value) in FP8_CACHES {
        let (prompt, decode) = (cached(512, 4096, key, value), cached(1, 4096, key, value));
        let reads = fragment_reads(&hw, &prompt);
        assert!(
            reads.iter().any(|(score, _)| *score == e4m3) == (key == E4M3)
                && reads.iter().any(|(_, read)| *read == e4m3) == (value == E4M3),
            "{key:?} keys, {value:?} values: the fragments read {reads:?}"
        );
        let widened = fragment_reads(&without, &prompt);
        assert!(
            !widened.is_empty()
                && widened
                    .iter()
                    .all(|read| *read == (f16.clone(), f16.clone())),
            "{key:?} keys, {value:?} values without an e4m3 instruction: the fragments read \
             {widened:?}"
        );

        for problem in [&prompt, &decode] {
            for (prepared, name) in [
                (
                    prepares(HypercubeShufflePartialAttentionRoutine::new(), &hw, problem),
                    "the split-keys partial shuffle",
                ),
                (
                    prepares(
                        HypercubeLastArrivalShuffleAttentionRoutine::new(),
                        &hw,
                        problem,
                    ),
                    "the last-arrival shuffle",
                ),
                (
                    prepares(CubeShuffleAttentionRoutine, &hw, problem),
                    "the cube shuffle",
                ),
            ] {
                assert!(
                    prepared,
                    "{name} declines {key:?} keys beside {value:?} values at {} queries",
                    problem.seq_q
                );
            }
        }
        assert!(
            !prepares(UnitAttentionRoutine, &hw, &decode),
            "the unit routine reads {key:?} keys beside {value:?} values"
        );
    }
}
