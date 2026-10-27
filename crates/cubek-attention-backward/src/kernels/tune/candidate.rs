//! The candidates the backward pass's autotune races.

use crate::kernels::AttentionBackwardProblem;
use crate::kernels::AttentionBackwardRoutine;
use crate::kernels::AttentionBackwardTensors;
use crate::kernels::tune::AttentionBackwardKey;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate, short_of_the_device};

/// One candidate of the backward pass's autotune: a routine at one of its settings, prepared for
/// the device the autotune is built for.
///
/// Its priority is how far the walk owned by the keys, the larger of the two, falls short of
/// filling the device ([`short_of_the_device`]).
#[derive(Debug, Clone)]
pub struct AttentionBackwardCandidate<R: AttentionBackwardRoutine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
}

impl<R: AttentionBackwardRoutine + Copy> AttentionBackwardCandidate<R> {
    /// Every setting `routine` lists on `hw`, as autotune candidates, its default first.
    pub(crate) fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine,
                settings,
                hw: hw.clone(),
            })
            .collect()
    }

    /// `tensors`' gradients written by this candidate's routine at its setting, or why it did not
    /// run.
    pub(crate) fn launch(&self, tensors: AttentionBackwardTensors) -> Result<(), String> {
        let hw = Hardware::from_client(tensors.client());
        self.routine
            .launch_on(&hw, tensors, &self.settings)
            .map(|_| ())
            .map_err(|why| format!("{} does not serve this problem: {why:?}", self.settings))
    }
}

impl<R: AttentionBackwardRoutine> TuneCandidate for AttentionBackwardCandidate<R> {
    type Strategy = R::Settings;
    type Key = AttentionBackwardKey;
    type Blueprint = ();

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the key walk's grid falls short of filling the
    /// device; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &AttentionBackwardKey) -> CandidatePriority {
        let prepared = AttentionBackwardProblem::try_from(key)
            .ok()
            .and_then(|problem| {
                self.routine
                    .prepare(&self.hw, &problem, &self.settings)
                    .ok()
            });
        match prepared {
            Some(blueprint) => CandidatePriority::FIRST.lowered(short_of_the_device(
                &self.hw,
                &blueprint.key_gradient.grid.launch,
            )),
            None => CandidatePriority::LAST,
        }
    }
}
