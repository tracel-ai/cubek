use super::ConvTuneKey;
use crate::kernels::tiled::{
    ConvProblem, ConvRoutine, ConvSettings, ConvTensors, CubeStagedConvRoutine, TileConvBlueprint,
};
use cubecl::CubeDim;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::routine::RoutineCandidate;

/// One listed setting, raced for the device it was listed on.
///
/// Its priority is read off the launch it prepares for the key, and that launch is what two
/// settings that snap onto one are told apart by.
#[derive(Debug, Clone)]
pub struct ConvCandidate {
    candidate: RoutineCandidate<CubeStagedConvRoutine>,
}

impl ConvCandidate {
    /// `settings` on `hw`, having prepared nothing yet.
    pub fn new(settings: ConvSettings, hw: &Hardware) -> Self {
        Self {
            candidate: RoutineCandidate::new(CubeStagedConvRoutine, settings, hw),
        }
    }

    /// Every setting the routine lists on `hw`, the first of them the base.
    pub fn list(hw: &Hardware) -> Vec<Self> {
        RoutineCandidate::list(CubeStagedConvRoutine, hw)
            .into_iter()
            .map(|candidate| Self { candidate })
            .collect()
    }

    /// The settings this candidate launches.
    pub fn settings(&self) -> &ConvSettings {
        self.candidate.settings()
    }

    /// The convolution `tensors` state, at this candidate's settings.
    ///
    /// # Errors
    ///
    /// Why the device cannot run it at these settings, as [`serves`](Self::serves) says.
    pub fn launch(&self, tensors: ConvTensors) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let prepared = self.candidate.prepare(&problem)?;
        CubeStagedConvRoutine.launch_prepared(&problem, tensors, prepared)
    }

    /// Whether the device runs `tensors` at these settings: prepared and bound, nothing launched.
    ///
    /// # Errors
    ///
    /// The refusal of the routine's preparation, or of the binding at a width its operands cannot
    /// serve.
    pub fn serves(&self, tensors: &ConvTensors) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let (blueprint, grid) = self.candidate.prepare(&problem)?;
        tensors.clone().bind(&problem, &blueprint, &grid).map(drop)
    }
}

impl TuneCandidate for ConvCandidate {
    type Strategy = ConvSettings;
    type Key = ConvTuneKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (TileConvBlueprint<CubeStagedMicrokernel>, CubeDim);

    fn strategy(&self) -> &ConvSettings {
        self.settings()
    }

    /// Last where the routine declines the problem, which it then does at launch before compiling
    /// anything.
    fn priority(&self, key: &ConvTuneKey) -> CandidatePriority {
        let problem = ConvProblem::from(key);
        match self.candidate.launch_of(&problem) {
            Some((blueprint, cube)) => blueprint.partitioning.priority(
                &problem,
                self.candidate.hardware(),
                cube.num_elems() as usize,
            ),
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &ConvTuneKey) -> Option<Self::Blueprint> {
        self.candidate.launch_of(&ConvProblem::from(key))
    }
}
