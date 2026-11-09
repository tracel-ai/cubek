//! The tune candidate the walk's autotune races.

use crate::kernels::walk::tune::key::GatedDeltaWalkKey;
use crate::kernels::walk::{
    GatedDeltaWalkBlueprint, GatedDeltaWalkInputs, GatedDeltaWalkOutputs, GatedDeltaWalkProblem,
    GatedDeltaWalkRoutine, GatedDeltaWalkTensors,
};
use cubecl::CubeDim;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate, short_of_the_device};
use cubek_std::routine::{Grid, PreparedLaunches, Routine};
use std::sync::Arc;

/// One candidate of the walk's autotune: a routine at one of its settings, prepared for the device
/// the autotune is built for.
///
/// Its priority is how far the prepared launch's grid falls short of filling the device
/// ([`short_of_the_device`]): a decode step of one sequence in cubes of a whole head is the shape
/// that leaves it idle.
#[derive(Debug, Clone)]
pub struct GatedDeltaWalkCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine at
    /// the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> GatedDeltaWalkCandidate<R> {
    /// `routine` at `settings`, on `hw`, having prepared nothing yet.
    pub fn new(routine: R, settings: R::Settings, hw: &Hardware) -> Self {
        Self {
            routine,
            settings,
            hw: hw.clone(),
            prepared: Arc::default(),
        }
    }
}

impl<R: Routine + Copy> GatedDeltaWalkCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates of the autotune, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }
}

impl<R, M> GatedDeltaWalkCandidate<R>
where
    R: GatedDeltaWalkRoutine<M> + Routine<Microkernel = M, Blueprint = GatedDeltaWalkBlueprint<M>>,
{
    /// The walk of `inputs` launched on its device at this candidate's setting, each problem
    /// prepared once ([`PreparedLaunches`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the walk at this setting.
    pub fn launch(&self, inputs: GatedDeltaWalkInputs) -> Result<GatedDeltaWalkOutputs, Refusal> {
        let tensors = GatedDeltaWalkTensors::new(inputs)?;
        let (blueprint, grid) =
            self.prepared
                .prepare(&self.routine, &self.hw, tensors.problem(), &self.settings)?;
        let binding = tensors.bind(&blueprint, &grid);
        self.routine.dispatch(&tensors, binding, blueprint, grid);
        Ok(tensors.into_outputs())
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device. Not held: a key's problem places the candidate in a
    /// round, and may never be launched.
    fn prepared(&self, key: &GatedDeltaWalkKey) -> Option<(GatedDeltaWalkBlueprint<M>, Grid)> {
        self.routine
            .prepare(&self.hw, &GatedDeltaWalkProblem::from(key), &self.settings)
            .ok()
    }

    pub(super) fn run(
        &self,
        inputs: GatedDeltaWalkInputs,
    ) -> Result<GatedDeltaWalkOutputs, String> {
        self.launch(inputs).map_err(|refusal| refusal.to_string())
    }
}

/// The candidate's name: its setting's, which is what a persisted winner is checksummed by.
impl<R: Routine<Settings: core::fmt::Display>> core::fmt::Display for GatedDeltaWalkCandidate<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl<R, M: PartialEq> TuneCandidate for GatedDeltaWalkCandidate<R>
where
    R: GatedDeltaWalkRoutine<M> + Routine<Microkernel = M, Blueprint = GatedDeltaWalkBlueprint<M>>,
{
    type Strategy = R::Settings;
    type Key = GatedDeltaWalkKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (GatedDeltaWalkBlueprint<M>, CubeDim);

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem.
    fn priority(&self, key: &GatedDeltaWalkKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &GatedDeltaWalkKey) -> Option<Self::Blueprint> {
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}
