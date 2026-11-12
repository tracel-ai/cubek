//! The candidates the backward pass's autotune races.

use crate::kernels::AttentionBackwardLaunch;
use crate::kernels::AttentionBackwardProblem;
use crate::kernels::AttentionBackwardTensors;
use crate::kernels::tune::AttentionBackwardKey;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate, short_of_the_device};

/// One candidate of the backward pass's autotune: a strategy at one of its settings, prepared for
/// the device the autotune is built for.
///
/// Its priority is how far the strategy's widest launch, the walk owned by the keys, falls short
/// of filling the device ([`short_of_the_device`]).
#[derive(Debug, Clone)]
pub struct AttentionBackwardCandidate<S: AttentionBackwardLaunch> {
    strategy: S,
    settings: S::Settings,
    hw: Hardware,
}

impl<S: AttentionBackwardLaunch + Copy> AttentionBackwardCandidate<S> {
    /// Every setting `strategy` lists on `hw`, as autotune candidates, its default first.
    pub(crate) fn list(strategy: S, hw: &Hardware) -> Vec<Self> {
        strategy
            .settings(hw)
            .into_iter()
            .map(|settings| Self {
                strategy,
                settings,
                hw: hw.clone(),
            })
            .collect()
    }

    /// `tensors`' gradients written by this candidate's strategy at its setting, or why it did
    /// not run.
    pub(crate) fn launch(&self, tensors: AttentionBackwardTensors) -> Result<(), String> {
        let hw = Hardware::from_client(tensors.client());
        self.strategy
            .launch_on(&hw, tensors, &self.settings)
            .map(|_| ())
            .map_err(|why| format!("{} does not serve this problem: {why:?}", self.settings))
    }
}

impl<S: AttentionBackwardLaunch> TuneCandidate for AttentionBackwardCandidate<S> {
    type Strategy = S::Settings;
    type Key = AttentionBackwardKey;
    type Blueprint = ();

    fn strategy(&self) -> &S::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the widest launch's grid falls short of filling
    /// the device; last where the strategy declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &AttentionBackwardKey) -> CandidatePriority {
        let prepared = AttentionBackwardProblem::try_from(key)
            .ok()
            .and_then(|problem| {
                self.strategy
                    .prepare(&self.hw, &problem, &self.settings)
                    .ok()
            });
        match prepared {
            Some(blueprint) => CandidatePriority::FIRST.lowered(short_of_the_device(
                &self.hw,
                &S::widest_grid(&blueprint).launch,
            )),
            None => CandidatePriority::LAST,
        }
    }
}
