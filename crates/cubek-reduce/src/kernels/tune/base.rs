use crate::ReduceError;
use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceKernel, ReduceRoutine, ReduceTensors, UnitDirectReduceRoutine,
    launch_kernel, tune::*,
};
use cubecl::client::Client;
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, TunableSet, TuneGroup, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{
    Autotune, CandidateIdentity, Ceiling, GroupPriority, Identity, Roofline, TuneCandidate,
};
use cubek_std::routine::EstimatedProblem;
use cubek_std::routine::Routine;
use std::sync::Arc;

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type ReduceIdentity = CandidateIdentity<ReduceCandidate>;

/// The autotune as it is built, before it closes into its candidates.
type ReduceAutotune = Autotune<ReduceKey, ReduceTensors, (), ReduceIdentity>;

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

/// `candidate`'s identity: its routine and its setting.
fn identity_of<R: Routine>(candidate: &ReduceRoutineRunner<R>) -> ReduceIdentity
where
    ReduceRoutineRunner<R>: Clone,
    ReduceCandidate: From<ReduceRoutineRunner<R>>,
{
    ReduceIdentity::Routine(ReduceCandidate::from(candidate.clone()))
}

/// `autotune` with every setting `R` lists on `hw` among `tile`'s candidates, each run on the
/// call's tensors.
fn racing<R, M>(
    autotune: ReduceAutotune,
    tile: &TuneGroup<ReduceKey>,
    hw: &Hardware,
) -> ReduceAutotune
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
    ReduceRoutineRunner<R>: TuneCandidate<Key = ReduceKey> + Clone + Send + Sync + 'static,
    ReduceCandidate: From<ReduceRoutineRunner<R>>,
{
    autotune.identified(
        tile,
        ReduceRoutineRunner::<R>::list(hw),
        identity_of,
        |candidate, tensors| candidate.run(tensors),
    )
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the unit
/// routine as the floor, which serves every problem on every device.
fn candidates(hw: Hardware) -> TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity> {
    let mut autotune = ReduceAutotune::new(|tensors: &ReduceTensors| ReduceKey::from(tensors));
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor_hw = hw.clone();
    let autotune = racing::<UnitDirectReduceRoutine, _>(autotune, &tile, &hw);
    let autotune = racing::<PlaneShuffleReduceRoutine, _>(autotune, &tile, &hw);
    let autotune = racing::<CubeStagedReduceRoutine, _>(autotune, &tile, &hw);
    let autotune = racing::<HypercubeLastArrivalReduceRoutine, _>(autotune, &tile, &hw);
    autotune
        .identified(
            &tile,
            ReduceRoutineRunner::<HypercubeAtomicReduceRoutine>::list(&hw),
            identity_of,
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
                launch_kernel(ReduceKernel::Unit, &floor_hw, &tensors)
                    .map_err(|decline| decline.to_string())
            },
        ))
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{
        BufferLaunch, BufferScopeChecker, given_too_little_scratch, sum_of_one_long_row,
    };
    use cubek_std::devices;
    use cubek_test_utils::ScopeChecker;

    const LAUNCH: BufferLaunch = |_, tensors| {
        launch_tuned(tensors.clone(), &cubecl::test_device())
            .expect("the floor serves every problem");
        Ok(())
    };

    strategy_reduces_each_problem!(LAUNCH);

    /// The hypercube declines a call given too little scratch.
    #[test]
    fn a_call_given_too_little_scratch_runs_another_routine() {
        const LAUNCH_SHORT: BufferLaunch =
            |hw, tensors| LAUNCH(hw, &given_too_little_scratch(tensors));
        BufferScopeChecker::new(LAUNCH_SHORT).check([sum_of_one_long_row()]);
    }

    /// Every setting of every routine, the floor and the groups' layout.
    #[test]
    fn every_setting_of_every_routine_races() {
        let hw = devices::m2_pro();
        let settings = UnitDirectReduceRoutine::settings(&hw).len()
            + PlaneShuffleReduceRoutine::settings(&hw).len()
            + CubeStagedReduceRoutine::settings(&hw).len()
            + HypercubeLastArrivalReduceRoutine::settings(&hw).len()
            + HypercubeAtomicReduceRoutine::settings(&hw).len();
        assert_eq!(candidates(hw).len(), settings + 2);
    }

    #[test]
    fn a_candidate_is_identified_by_its_routine_and_setting() {
        let runner =
            ReduceRoutineRunner::<UnitDirectReduceRoutine>::list(&devices::m2_pro()).remove(0);
        assert_eq!(identity_of(&runner).to_string(), *runner.strategy());
    }
}
