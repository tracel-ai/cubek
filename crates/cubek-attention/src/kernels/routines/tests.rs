//! The routines' candidate lists reach every value autotune needs on each lever: the step of
//! keys, the box of queries, the cuts of the keys across cubes and planes.

use crate::kernels::test_support::{offering, qwen};
use crate::kernels::*;
use cubecl::tile::stage::Prefetch;
use cubek_microkernel::attention::output::cube::CubeAttentionMicrokernelSettings;
use cubek_microkernel::attention::output::plane::{
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
