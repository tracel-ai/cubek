use super::{ConvCandidate, ConvTuneKey};
use crate::kernels::tiled::{ConvRoutine, ConvSettings, ConvTensors, UnitRegistersConvRoutine};
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

static TUNER: LocalTuner<ConvTuneKey, ServiceId> = local_tuner!("tile-conv");

/// The list each device races, built once so every launch and the autotune share the launches its
/// candidates prepared.
static LISTS: LazyLock<Mutex<HashMap<ServiceId, Arc<[ConvCandidate]>>>> =
    LazyLock::new(Mutex::default);

fn listed(id: &ServiceId, hw: &Hardware) -> Arc<[ConvCandidate]> {
    LISTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(*id)
        .or_insert_with(|| ConvCandidate::list(hw).into())
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
    let key = ConvTuneKey::of(&tensors.problem(), hw);
    let serves = |candidates: &mut dyn Iterator<Item = &ConvCandidate>| {
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
        true => {
            TUNER.execute(&id, &client, tunables, tensors);
            Ok(())
        }
        false => launch_floor(&list, hw, tensors),
    }
}

/// `tensors` at the first of `list` that serves them, or on the unit registers routine where none
/// does.
fn launch_floor(
    list: &[ConvCandidate],
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
    list: Arc<[ConvCandidate]>,
    hw: Hardware,
) -> TunableSet<ConvTuneKey, ConvTensors, (), CandidateIdentity<ConvSettings>> {
    let keyed = hw.clone();
    let mut autotune =
        Autotune::<ConvTuneKey, ConvTensors, (), CandidateIdentity<ConvSettings>>::new(
            move |tensors: &ConvTensors| ConvTuneKey::of(&tensors.problem(), &keyed),
        );
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor = (list.clone(), hw.clone());
    autotune
        .bounded(move |key: &ConvTuneKey, tensors: &ConvTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: key.ceiling(&hw),
            work: tensors.problem().demand(),
        })
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
        .floor(<CandidateIdentity<ConvSettings> as Identity>::named(
            "conv-tile-floor",
            move |tensors: ConvTensors| {
                let (list, hw) = &floor;
                launch_floor(list, hw, tensors).map_err(|refusal| format!("{refusal:?}"))
            },
        ))
        .finish()
}
