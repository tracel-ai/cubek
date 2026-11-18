use super::{CubeStagedConvCandidate, CubeStagedConvTuneKey};
use crate::kernels::tiled::{
    ConvRoutine, ConvTensors, CubeStagedConvSettings, UnitRegistersConvRoutine,
};
use cubecl::client::Client;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl_common::device::ServiceId;
use cubek_std::Hardware;
use cubek_std::autotune::{
    Autotune, CandidateIdentity, GroupPriority, Identity, Roofline, TuneCandidate,
};
use cubek_std::routine::{EstimatedProblem, ROUTINE_LISTS_NO_SETTING, first_served};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

const TILE_GROUP: &str = "tile";
const FLOOR_NAME: &str = "conv-tile-floor";

static TUNER: LocalTuner<CubeStagedConvTuneKey, ServiceId> = local_tuner!("tile-conv");

/// The list each device races, built once so every launch and the autotune share the launches its
/// candidates prepared.
static LISTS: LazyLock<Mutex<HashMap<ServiceId, Arc<[CubeStagedConvCandidate]>>>> =
    LazyLock::new(Mutex::default);

fn listed(id: &ServiceId, hw: &Hardware) -> Arc<[CubeStagedConvCandidate]> {
    LISTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(*id)
        .or_insert_with(|| CubeStagedConvCandidate::list(hw).into())
        .clone()
}

/// The convolution `tensors` state, at the setting the autotune finds fastest for it among the
/// routine's listed settings on `hw`.
///
/// Where no listed setting serves the problem, or the one the autotune settled for its key does
/// not, the first that does launches, and the unit registers routine is the floor under them all.
///
/// # Errors
///
/// The unit registers routine's refusal where nothing serves the problem.
pub fn launch_tuned(hw: &Hardware, tensors: ConvTensors) -> Result<(), Refusal> {
    let client = tensors.client().clone();
    let id = client.service_id();
    let list = listed(&id, hw);
    let tunables = TUNER.init(&id, {
        let (list, hw) = (list.clone(), hw.clone());
        move || candidates(list.clone(), hw.clone())
    });
    let key = CubeStagedConvTuneKey::of(&tensors.problem(), hw);
    let serves = |candidates: &mut dyn Iterator<Item = &CubeStagedConvCandidate>| {
        let attempts = candidates.map(|candidate| candidate.serves(&tensors));
        first_served(attempts, ROUTINE_LISTS_NO_SETTING).is_ok()
    };
    let tuned = match TUNER.fastest_identity(&id, &tunables, &key) {
        None => serves(&mut list.iter()),
        Some(winner) => match winner.routine() {
            Some(settings) => serves(&mut list.iter().filter(|c| c.settings() == &settings)),
            None => true,
        },
    };
    match tuned {
        true => run_tuned(&id, &client, tunables, tensors),
        false => launch_floor(&list, hw, tensors),
    }
}

/// `tensors` launched at the setting the autotune picks among `tunables`.
fn run_tuned(
    id: &ServiceId,
    client: &Client,
    tunables: Arc<
        TunableSet<
            CubeStagedConvTuneKey,
            ConvTensors,
            (),
            CandidateIdentity<CubeStagedConvSettings>,
        >,
    >,
    tensors: ConvTensors,
) -> Result<(), Refusal> {
    TUNER.execute(id, client, tunables, tensors);
    Ok(())
}

/// `tensors` at the first of `list` that serves them, or on the unit registers routine where none
/// does.
fn launch_floor(
    list: &[CubeStagedConvCandidate],
    hw: &Hardware,
    tensors: ConvTensors,
) -> Result<(), Refusal> {
    let listed = list
        .iter()
        .map(|candidate| candidate.launch(tensors.clone()));
    let floor =
        std::iter::once_with(|| UnitRegistersConvRoutine.launch_first_served(hw, tensors.clone()));
    first_served(listed.chain(floor), ROUTINE_LISTS_NO_SETTING)
}

/// The autotune over `list` on `hw`: every listed setting, a round stopping at the first within
/// reach of the bound the convolution has, and [`launch_floor`] where none serves.
pub(super) fn candidates(
    list: Arc<[CubeStagedConvCandidate]>,
    hw: Hardware,
) -> TunableSet<CubeStagedConvTuneKey, ConvTensors, (), CandidateIdentity<CubeStagedConvSettings>> {
    let keyed = hw.clone();
    let mut autotune = Autotune::<
        CubeStagedConvTuneKey,
        ConvTensors,
        (),
        CandidateIdentity<CubeStagedConvSettings>,
    >::new(move |tensors: &ConvTensors| {
        CubeStagedConvTuneKey::of(&tensors.problem(), &keyed)
    });
    let tile = autotune.group(TILE_GROUP, GroupPriority::DEFAULT);
    let floor = (list.clone(), hw.clone());
    autotune
        .bounded(
            move |key: &CubeStagedConvTuneKey, tensors: &ConvTensors| Roofline {
                client: tensors.client().clone(),
                ceiling: key.ceiling(&hw),
                work: tensors.problem().demand(),
            },
        )
        .identified(
            &tile,
            list.to_vec(),
            CandidateIdentity::new,
            |candidate, tensors| {
                candidate.launch(tensors).map_err(|refusal| {
                    format!(
                        "{} does not serve this problem: {refusal:?}",
                        candidate.strategy()
                    )
                })
            },
        )
        .floor(
            <CandidateIdentity<CubeStagedConvSettings> as Identity>::named(
                FLOOR_NAME,
                move |tensors: ConvTensors| {
                    let (list, hw) = &floor;
                    launch_floor(list, hw, tensors).map_err(|refusal| format!("{refusal:?}"))
                },
            ),
        )
        .finish()
}
