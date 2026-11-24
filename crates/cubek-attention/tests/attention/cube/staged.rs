//! The cube staged routine: a cube's planes of fragments contracting the stages it fills.

use core::num::NonZeroUsize;
use cubek_attention::kernels::{CubeStagedAttentionRoutine, CubeStagedAttentionRoutineSettings};
use cubek_microkernel::attention::CacheLines;
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernelSettings, CubeAttentionPlaneSettings,
};
use cubek_microkernel::attention::plane::{PlaneFragmentAttentionMicrokernelSettings, QueryHolder};
use cubek_std::microkernel::{StageLayout, StageRing};
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{
    Forward, PROMPT, PROMPT_HEAD_DIM_256, PROMPT_WIDE, PROMPTS, PROMPTS_HEAD_DIM_256_PAGED,
    PROMPTS_WIDE_FP8, PROMPTS_WIDE_PAGED, PROMPTS_WIDE_PER_TENSOR,
};

/// The cube staged routine's lead on every prompt case.
#[test]
fn the_cube_staged_routine_matches_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS));
}

/// Planes that split the same rows' keys merge their state before they write it.
#[test]
fn planes_splitting_the_keys_match_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    let split = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.microkernel.planes_along_keys.get() > 1)
        .take(1);
    forward.check(forward.at(split, &[PROMPT]));
}

/// The lead's box in swizzled stages at a model's head dim. Where the device offers no manual
/// `mma` the routine lists no swizzled stage: its cmma fragments read the stage padded, which
/// this checks against the host.
#[test]
fn a_swizzled_stage_matches_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    let swizzled =
        forward
            .settings()
            .into_iter()
            .take(1)
            .map(|lead| CubeStagedAttentionRoutineSettings {
                microkernel: CubeAttentionMicrokernelSettings {
                    layout: StageLayout::SwizzledBlocks,
                    ..lead.microkernel
                },
                ..lead
            });
    forward.check(forward.at(swizzled, &[PROMPT_WIDE]));
}

/// Planes holding their query in registers: one plane along the queries and several, over one and
/// several key fragments a step, with the next stage in a second slot and waiting in registers, the
/// values staged as they are stored and along the keys. A device that keeps no score in fragments,
/// or whose shared memory does not hold the stages, declines.
fn held_queries() -> [CubeStagedAttentionRoutineSettings; 5] {
    let through_registers = StageRing::SINGLE_THROUGH_REGISTERS;
    [
        (1, 1, 1, StageRing::SINGLE, CacheLines::AsStored),
        (2, 2, 1, StageRing::DOUBLE, CacheLines::AsStored),
        (6, 4, 1, through_registers, CacheLines::AlongKeys),
        (6, 4, 1, StageRing::DOUBLE, CacheLines::AlongKeys),
        (2, 4, 2, StageRing::SINGLE, CacheLines::AlongKeys),
    ]
    .map(|(heads, planes, keys, ring, values)| {
        let lead = CubeStagedAttentionRoutineSettings::DEFAULT;
        CubeStagedAttentionRoutineSettings {
            microkernel: CubeAttentionMicrokernelSettings {
                plane: CubeAttentionPlaneSettings::Fragment(
                    PlaneFragmentAttentionMicrokernelSettings::fragments(1, keys)
                        .querying(QueryHolder::Registers)
                        .along(values),
                ),
                heads: NonZeroUsize::new(heads).unwrap(),
                planes_along_queries: NonZeroUsize::new(planes).unwrap(),
                ring,
                layout: StageLayout::PaddedRows,
                ..lead.microkernel
            },
            ..lead
        }
    })
}

/// A held query reads its values from global memory once, past the last query a box cuts, and
/// casts them as they load beside `e4m3` keys, the probabilities cast to `e4m3` values' own.
#[test]
fn a_query_held_in_registers_matches_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at(held_queries(), &[PROMPT_WIDE, PROMPT_HEAD_DIM_256]));
    forward.check(forward.at(held_queries(), &PROMPTS_WIDE_FP8));
}

/// Values staged along the keys beside a query staged with them.
#[test]
fn values_along_the_keys_beside_a_staged_query_match_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    let lead = CubeStagedAttentionRoutineSettings::DEFAULT;
    let along_keys = CubeStagedAttentionRoutineSettings {
        microkernel: CubeAttentionMicrokernelSettings {
            plane: CubeAttentionPlaneSettings::Fragment(
                PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1)
                    .along(CacheLines::AlongKeys),
            ),
            layout: StageLayout::PaddedRows,
            ..lead.microkernel
        },
        ..lead
    };
    forward.check(forward.at([along_keys], &[PROMPT_WIDE]));
    forward.check(forward.at([along_keys], &PROMPTS_WIDE_FP8));
}

/// A held query over a cache in pages, `f16` and `e4m3`: a stage fills from the page its table
/// names, each side in the lines it is staged along.
#[test]
fn a_query_held_in_registers_reads_a_paged_cache() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at(held_queries(), &PROMPTS_WIDE_PAGED));
    forward.check(forward.at(held_queries(), &PROMPTS_HEAD_DIM_256_PAGED));
}

/// The cube staged routine's lead over a cache in pages.
#[test]
fn the_cube_staged_routine_reads_a_paged_cache() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS_WIDE_PAGED));
}

/// Every setting the routine lists at a head dim of 256, where the output's rows a correction
/// scales run sixteen fragments wide: a setting whose stages the device's shared memory does not
/// hold declines.
#[test]
fn every_cube_staged_setting_matches_the_host_at_a_head_dim_of_256() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_HEAD_DIM_256]));
}

/// Every setting the routine lists, the copy engine's among them, at a model's head dim: the
/// query's and the stages' layouts, the rings and the run of score fragments the value reads are
/// each a different kernel.
#[test]
fn every_cube_staged_setting_matches_the_host_at_a_head_dim_of_128() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_WIDE]));
}

/// Every setting the routine lists over every FP8 cache at a model's head dim: the score on the
/// device's FP8 instruction, the `f16` query converted ahead, and `e4m3` values multiplied on it
/// beside the probabilities cast to `e4m3`, or widened to the query's element by their stage.
#[test]
fn every_cube_staged_setting_reads_every_fp8_cache() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_FP8));
}

/// Every setting the routine lists over a cache quantized per tensor at a model's head dim: the
/// stages hold the keys and values as stored, the softmax reads the score under the keys' scale,
/// and the output drains under the values'.
#[test]
fn every_cube_staged_setting_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_PER_TENSOR));
}
