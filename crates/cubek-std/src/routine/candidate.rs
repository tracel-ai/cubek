use crate::Hardware;
use crate::routine::{Grid, PreparedLaunches, Routine};
use cubecl::CubeDim;
use cubecl::tile::launch::Refusal;
use std::sync::Arc;

/// A routine at one of its settings, raced for the device it was listed on.
///
/// The launches it prepares are held ([`PreparedLaunches`]), so a clone, and the autotune that
/// ranks it and the launch that runs it, prepare each problem once.
#[derive(Debug, Clone)]
pub struct RoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> RoutineCandidate<R> {
    /// `routine` at `settings` on `hw`, having prepared nothing yet.
    pub fn new(routine: R, settings: R::Settings, hw: &Hardware) -> Self {
        Self {
            routine,
            settings,
            hw: hw.clone(),
            prepared: Arc::default(),
        }
    }

    /// The setting this candidate launches the routine at.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }

    /// The device it was listed for.
    pub fn hardware(&self) -> &Hardware {
        &self.hw
    }
}

impl<R: Routine + Copy> RoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }
}

impl<R: Routine> RoutineCandidate<R>
where
    R::Problem: PartialEq,
{
    /// The launch the routine prepares for `problem` at this setting, each problem prepared once.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run `problem` at this setting, as [`Routine::prepare`] says.
    pub fn prepare(&self, problem: &R::Problem) -> Result<(R::Blueprint, Grid), Refusal> {
        self.prepared
            .prepare(&self.routine, &self.hw, problem, &self.settings)
    }

    /// The blueprint the routine prepares for `problem` and the cube it launches, or `None`
    /// where the routine declines it. The cube is kept out of a blueprint, and two settings
    /// preparing one blueprint can still launch different cubes.
    pub fn launch_of(&self, problem: &R::Problem) -> Option<(R::Blueprint, CubeDim)> {
        let (blueprint, grid) = self.prepare(problem).ok()?;
        Some((blueprint, grid.launch.cube_dim))
    }
}
