//! The launch a [`SwigluRoutineCandidate`](crate::kernels::SwigluRoutineCandidate) prepares for a
//! key ([`SwigluRoutinePrepared`]).

use crate::kernels::SwigluBlueprint;
use cubek_std::routine::{Grid, LaunchSettings};

/// The launch a routine at one setting prepares for a key: the grid it runs on, and what it
/// compiles, which two candidates compile alike exactly where their prepared launches are equal.
#[derive(Clone, Debug)]
pub struct SwigluRoutinePrepared<M> {
    blueprint: SwigluBlueprint<M>,
    launch: LaunchSettings,
}

impl<M> SwigluRoutinePrepared<M> {
    /// The cubes and units the launch spawns.
    pub fn launch(&self) -> &LaunchSettings {
        &self.launch
    }
}

impl<M> SwigluRoutinePrepared<M> {
    /// The launch of `blueprint` on `grid`.
    pub(crate) fn new(blueprint: SwigluBlueprint<M>, grid: Grid) -> Self {
        Self {
            blueprint,
            launch: grid.launch,
        }
    }
}

/// Two launches compile alike where their blueprint and cube do: the cube is kept out of a
/// blueprint, two settings preparing one blueprint can still launch different cubes, and the count
/// of cubes is a runtime value.
impl<M: PartialEq> PartialEq for SwigluRoutinePrepared<M> {
    fn eq(&self, other: &Self) -> bool {
        let cube = |prepared: &Self| prepared.launch.cube_dim;
        self.blueprint == other.blueprint && cube(self) == cube(other)
    }
}
