//! Choosing an attention's routine by measurement, in one of two autotunes: a call of many query
//! positions, and a call of one.
//!
//! The routines serve GQA natively, each key head read once per box. They attend the first
//! `len` positions of a cache with the queries bottom-right aligned, so a call over a cache is
//! raced on the one chunk, or the one decode step, every key over a cache stands for
//! ([`AttentionKey`]): no key reads how many positions a cache holds.

use crate::kernels::plan::{AttentionCall, AttentionPlan};
use crate::kernels::select::problem::QueryPositions;
use crate::kernels::tune::AttentionTuneInputs;
use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{
    AttentionRoutine, AttentionRoutineCandidate, CubeShuffleAttentionRoutine,
    CubeStagedAttentionRoutine, HypercubeLastArrivalShuffleAttentionRoutine,
    HypercubeSecondLaunchFragmentAttentionRoutine, HypercubeSecondLaunchShuffleAttentionRoutine,
    HypercubeStreamFragmentAttentionRoutine, UnitAttentionRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};
use cubek_std::plan::Fastest;

/// The priority each autotune's routines ship at: a caller's to set, as in the matmul family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionAutotuneGroups {
    /// The many-positions autotune's.
    pub prefill: GroupPriority,
    /// The one-position autotune's.
    pub decode: GroupPriority,
}

impl AttentionAutotuneGroups {
    /// The priority of the autotune of calls of `positions` query positions.
    pub fn of(self, positions: QueryPositions) -> GroupPriority {
        match positions {
            QueryPositions::One => self.decode,
            QueryPositions::Many => self.prefill,
        }
    }
}

impl Default for AttentionAutotuneGroups {
    fn default() -> Self {
        Self {
            prefill: GroupPriority::DEFAULT,
            decode: GroupPriority::DEFAULT,
        }
    }
}

/// The autotune of calls of `positions` query positions as a plan, its routines' group at
/// `routines`: the decode autotune at one position, the prefill one at many. It measures a float
/// call; a packed cache is rebuilt before it.
pub fn fastest(positions: QueryPositions, routines: GroupPriority) -> AttentionPlan {
    let autotune = match positions {
        QueryPositions::One => "decode",
        QueryPositions::Many => "prefill",
    };
    Box::new(Fastest::new(
        autotune,
        move |call: &AttentionCall| match call {
            AttentionCall::Float(float) => {
                launch_tuned(positions, &float.device, float.inputs.clone(), routines)
            }
            AttentionCall::Packed(_) => {
                unreachable!(
                    "an autotune is planned over a float call, the packed cache rebuilt first"
                )
            }
        },
    ))
}

/// The attention of `query` over `key` and `value`, `[b, q, h, dv]`, through whichever routine
/// and setting measures fastest for this problem on `device`: among the routines that serve many
/// query positions, or those that serve one, as `positions` says. `routines` is the priority of
/// the routines' group; under it sits the unit routine's lead, which serves any problem on any
/// device.
pub fn launch_tuned(
    positions: QueryPositions,
    device: &Device,
    inputs: AttentionTuneInputs,
    routines: GroupPriority,
) -> TensorHandle {
    let client = inputs.client.clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    match positions {
        QueryPositions::Many => {
            // Named rather than `module_path!()`, so the cached winners stay this autotune's
            // wherever the code lives; a cached winner is an index into the candidate list.
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("prefill");
            let tunables = TUNER.init(&id, move || many_queries(&hw, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
        QueryPositions::One => {
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("decode");
            let tunables = TUNER.init(&id, move || one_query(&hw, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
    }
    inputs.tensors.into_out()
}

type AttentionAutotune = Autotune<AttentionKey, AttentionTuneInputs, ()>;

/// The routines that serve many query positions: the cube routines, and the hypercube routines for
/// a prompt's or a chunk's few boxes over many keys. The unit routine is listed only where none of
/// them can run, for want of a plane width the device commits to: elsewhere measuring its
/// candidates takes longer than the rest of the autotune. The plane routine is not listed.
///
/// Its candidates are timed as the cache leaves them, unlike one query's: a prompt's queries, keys
/// and values are the activations the layer before it just wrote, so production reads them warm,
/// and a round timing them cold ranks the candidates on a read they never meet (an f32 head dim
/// of 128 on an L4: a winner 14% behind the multi-level kernels warm, against 28% chosen cold).
fn many_queries(
    hw: &Hardware,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::measured_on(
        AttentionTuneInputs::many_queries_key,
        "as-keyed",
        AttentionTuneInputs::many_queries_representative,
    );
    let routines = autotune.group("microkernel", priority);
    autotune = autotune.prioritized(
        &routines,
        AttentionRoutineCandidate::list(CubeStagedAttentionRoutine, hw),
        AttentionRoutineCandidate::launch,
    );
    autotune = autotune
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(CubeShuffleAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                HypercubeSecondLaunchShuffleAttentionRoutine::new(),
                hw,
            ),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeSecondLaunchFragmentAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeStreamFragmentAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        );
    if hw.plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionRoutineCandidate::list(UnitAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        );
    }
    autotune.floor(unit_lead(hw)).finish()
}

/// The routines that serve a single position, reading its group of heads as rows. The
/// hypercube's cubes of plane shuffles share each key head's cache, merged on the last arrival or
/// by a second launch; the cube's plane shuffles split a cube's keys. No routine whose cubes stage
/// is listed: a cube staging a step for a decode step's few rows measured twelve times the
/// hypercube's time. As for many positions, the unit routine is listed only where neither can
/// run, and the plane routine not at all.
fn one_query(
    hw: &Hardware,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::measured_on(
        AttentionTuneInputs::one_query_key,
        "as-keyed",
        AttentionTuneInputs::one_query_representative,
    );
    let routines = autotune.group("microkernel", priority);
    autotune = autotune
        .read_cold(AttentionTuneInputs::one_query_roofline)
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                HypercubeSecondLaunchShuffleAttentionRoutine::new(),
                hw,
            ),
            AttentionRoutineCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(CubeShuffleAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        );
    if hw.plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionRoutineCandidate::list(UnitAttentionRoutine, hw),
            AttentionRoutineCandidate::launch,
        );
    }
    autotune.floor(unit_lead(hw)).finish()
}

/// Both autotunes' floor: the unit routine's lead, which asks nothing of the hardware. Nobody
/// wants it to win, but a plan cubecl cannot fill is a panic, and a configuration may take every
/// group out of the autotune.
fn unit_lead(hw: &Hardware) -> Tunable<AttentionKey, AttentionTuneInputs, ()> {
    let hw = hw.clone();
    Tunable::new("unit-attention-lead", move |inputs: AttentionTuneInputs| {
        UnitAttentionRoutine
            .launch_lead_on(&hw, inputs.tensors)
            .map(|_| ())
            .map_err(|why| format!("the unit routine's lead does not serve this problem: {why:?}"))
    })
}
