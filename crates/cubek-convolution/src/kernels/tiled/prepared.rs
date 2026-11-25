use crate::kernels::tiled::{ConvCoverage, ConvProblem, TileConvBlueprint};
use cubecl::CubeDim;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;

/// The launch a [`CubeStagedConvCandidate`](crate::kernels::tiled::CubeStagedConvCandidate)
/// prepares for a key: how its boxes lie over the problem, the cube it launches, and what it
/// compiles.
///
/// # Guarantees
///
/// Two prepared launches are equal exactly where they compile one kernel and launch it in one
/// cube, however their boxes lie over the problem.
#[derive(Debug, Clone)]
pub struct CubeStagedConvPrepared {
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    compiled: (TileConvBlueprint<CubeStagedMicrokernel>, CubeDim),
    coverage: ConvCoverage,
}

impl CubeStagedConvPrepared {
    /// How the launch's boxes lie over the problem.
    pub fn coverage(&self) -> &ConvCoverage {
        &self.coverage
    }

    /// The units each cube of the launch spawns.
    pub fn cube_dim(&self) -> CubeDim {
        self.compiled.1
    }
}

impl CubeStagedConvPrepared {
    /// The launch of `blueprint` in cubes of `cube_dim` over `problem`.
    pub(crate) fn new(
        problem: &ConvProblem,
        blueprint: TileConvBlueprint<CubeStagedMicrokernel>,
        cube_dim: CubeDim,
    ) -> Self {
        Self {
            coverage: blueprint.partitioning.coverage(problem),
            compiled: (blueprint, cube_dim),
        }
    }
}

/// Two launches compile alike where their blueprint and cube do.
impl PartialEq for CubeStagedConvPrepared {
    fn eq(&self, other: &Self) -> bool {
        self.compiled == other.compiled
    }
}
