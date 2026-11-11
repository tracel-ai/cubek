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
                match HypercubeSecondLaunchFragmentAttentionRoutine.prepare(&hw, &problem, settings)
                {
                    Ok(_) => {}
                    Err(why) if position > 0 && why == SWIZZLED_STAGE_UNREAD => {}
                    prepared => panic!("{settings} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }
}

/// A prompt whose query and keys are `e4m3` beside `f16` values: an FP8 score.
fn fp8_score(queries: usize, keys: usize) -> AttentionRoutineProblem {
    let e4m3 = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::E4M3);
    let f16 = qwen(queries, keys);
    AttentionRoutineProblem {
        elements: AttentionOperands {
            query: e4m3,
            key: e4m3,
            ..f16.elements
        },
        ..f16
    }
}

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

/// An FP8 score is the plane fragments' alone: their score on the device's `e4m3` instruction,
/// their value on its 16-bit one. The shuffles and the unit routine, which compute in their
/// operands' one element, decline it, and so do the fragments on a device without the `e4m3`
/// instruction.
#[test]
fn an_fp8_score_is_read_by_the_plane_fragments_alone() {
    let (hw, problem) = (offering_fp8(), fp8_score(512, 4096));
    let e4m3 = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::E4M3);
    let f16 = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16);
    let fragments = HypercubeSecondLaunchFragmentAttentionRoutine;
    let prepared: Vec<_> = fragments
        .settings(&hw)
        .iter()
        .filter_map(|settings| fragments.prepare(&hw, &problem, settings).ok())
        .collect();
    assert!(
        !prepared.is_empty(),
        "no fragment setting reads an FP8 score"
    );
    for (blueprint, _) in &prepared {
        let served = blueprint.microkernel.served(&problem.work_item()).elements;
        assert_eq!((served.query, served.key, served.value), (e4m3, e4m3, f16));
    }

    let declines = |prepared: bool, name: &str| assert!(!prepared, "{name} reads an FP8 score");
    let shuffle = HypercubeSecondLaunchShuffleAttentionRoutine::new();
    declines(
        shuffle
            .settings(&hw)
            .iter()
            .any(|settings| shuffle.prepare(&hw, &problem, settings).is_ok()),
        "the plane shuffle",
    );
    declines(
        UnitAttentionRoutine.settings(&hw).iter().any(|settings| {
            UnitAttentionRoutine
                .prepare(&hw, &problem, settings)
                .is_ok()
        }),
        "the unit routine",
    );
    let without = offering(devices::rtx_pro_6000(), 16);
    declines(
        fragments
            .settings(&without)
            .iter()
            .any(|settings| fragments.prepare(&without, &problem, settings).is_ok()),
        "the plane fragments without an e4m3 instruction",
    );
}
