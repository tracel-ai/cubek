//! A strategy of the backward pass at one of its settings ([`AttentionBackwardCandidate`]), as a
//! caller that measures the strategies lists and launches it.

use crate::kernels::{
    AttentionBackwardKey, AttentionBackwardLaunch, AttentionBackwardProblem,
    AttentionBackwardTensors,
};
use cubek_std::Hardware;
use cubek_std::routine::LaunchSettings;

/// A strategy of the backward pass at one of its settings, prepared for the device it was listed
/// for: what a caller that measures the strategies races.
#[derive(Debug, Clone)]
pub struct AttentionBackwardCandidate<S: AttentionBackwardLaunch> {
    strategy: S,
    settings: S::Settings,
    hw: Hardware,
}

impl<S: AttentionBackwardLaunch + Copy> AttentionBackwardCandidate<S> {
    /// Every setting `strategy` lists on `hw`, as candidates, its default first.
    pub fn list(strategy: S, hw: &Hardware) -> Vec<Self> {
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
}

impl<S: AttentionBackwardLaunch> AttentionBackwardCandidate<S> {
    /// The setting this candidate launches its strategy at, which names it.
    pub fn settings(&self) -> &S::Settings {
        &self.settings
    }

    /// `tensors`' gradients written by this candidate's strategy at its setting, on their client's
    /// device.
    ///
    /// # Errors
    ///
    /// Why the strategy does not serve the problem at its setting.
    pub fn launch(&self, tensors: AttentionBackwardTensors) -> Result<(), String> {
        let hw = Hardware::from_client(tensors.client());
        self.strategy
            .launch_on(&hw, tensors, &self.settings)
            .map(|_| ())
            .map_err(|why| format!("{} does not serve this problem: {why:?}", self.settings))
    }

    /// The cubes and units of the widest launch, the walk owned by the keys, this candidate
    /// prepares for the problem `key` describes on the device it was listed for, or `None` where
    /// the strategy declines it.
    pub fn widest_launch(&self, key: &AttentionBackwardKey) -> Option<LaunchSettings> {
        let problem = AttentionBackwardProblem::try_from(key).ok()?;
        self.strategy
            .prepare(&self.hw, &problem, &self.settings)
            .ok()
            .map(|blueprint| S::widest_grid(&blueprint).launch.clone())
    }
}
