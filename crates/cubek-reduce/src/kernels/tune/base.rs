use crate::ReduceError;
use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceKernel, ReduceTensors, UnitDirectReduceRoutine, tune::*,
};
use cubecl::client::Client;
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{
    Autotune, CandidateIdentity, Ceiling, GroupPriority, Identity, Roofline,
};
use cubek_std::routine::EstimatedProblem;
use std::sync::Arc;

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type ReduceIdentity = CandidateIdentity<ReduceCandidate>;

/// The autotune's candidates, as the tuner holds them.
type ReduceTunables = Arc<TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity>>;

static TUNER: LocalTuner<ReduceKey, CubeTuneId> = local_tuner!("reduce");

/// `tensors` reduced by the routine the autotune finds fastest for them on `device`. It never
/// refuses: the floor serves every problem.
pub(crate) fn launch_tuned(tensors: ReduceTensors, device: &Device) -> Result<(), ReduceError> {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client);
    TUNER.execute(&id, &client, tunables, tensors);
    Ok(())
}

/// The candidate the tuner on `client`'s `device` found fastest for `key`, if a round settled it
/// ([`ReduceCandidate::fastest`]).
pub(super) fn fastest_tuned(
    client: &Client,
    device: &Device,
    key: &ReduceKey,
) -> Option<ReduceCandidate> {
    let id = CubeTuneId::new(client, device);
    let tunables = tunables(&id, client);
    TUNER.fastest_identity(&id, &tunables, key)?.routine()
}

/// The autotune for the device `id` names, built on first touch. The one place it is built: a
/// tuner caches a set per initializer *type*, so a second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client) -> ReduceTunables {
    let hw = Hardware::from_client(client);
    TUNER.init(id, move || candidates(hw.clone()))
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the unit
/// routine as the floor, which serves every problem on every device.
fn candidates(hw: Hardware) -> TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity> {
    let mut autotune =
        Autotune::<ReduceKey, ReduceTensors, (), ReduceIdentity>::new(|tensors: &ReduceTensors| {
            ReduceKey::from(tensors)
        });
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor_hw = hw.clone();
    autotune
        .identified(
            &tile,
            ReduceRoutineRunner::<UnitDirectReduceRoutine>::list(&hw),
            |candidate| ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone())),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineRunner::<PlaneShuffleReduceRoutine>::list(&hw),
            |candidate| ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone())),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineRunner::<CubeStagedReduceRoutine>::list(&hw),
            |candidate| ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone())),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineRunner::<HypercubeLastArrivalReduceRoutine>::list(&hw),
            |candidate| ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone())),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineRunner::<HypercubeAtomicReduceRoutine>::list(&hw),
            |candidate| ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone())),
            |candidate, tensors| candidate.run_zeroed(tensors),
        )
        .bounded(|_, tensors: &ReduceTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: tensors.problem().demand(),
        })
        .floor(ReduceIdentity::named(
            "unit-direct-reduce floor",
            move |tensors: ReduceTensors| {
                ReduceKernel::Unit
                    .launch(&floor_hw, &tensors)
                    .map_err(|decline| decline.to_string())
            },
        ))
        .finish()
}
