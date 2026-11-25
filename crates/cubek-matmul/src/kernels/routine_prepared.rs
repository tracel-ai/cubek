//! The launch a [`MatmulRoutineCandidate`](crate::kernels::MatmulRoutineCandidate) prepares for a
//! key ([`MatmulRoutinePrepared`]).

use crate::kernels::{CubeBlock, MatmulBlueprint};
use cubecl::CubeDim;
use cubek_std::matmul_dimensions::{K, M, N};

/// The launch a float routine at one setting prepares for a key: what it compiles, which two
/// candidates compile alike exactly where their prepared launches are equal, and the output block
/// one of its cubes covers.
#[derive(Clone, Debug)]
pub struct MatmulRoutinePrepared<Mk, SM> {
    blueprint: MatmulBlueprint<Mk, SM>,
    cube_dim: CubeDim,
}

impl<Mk, SM> MatmulRoutinePrepared<Mk, SM> {
    /// What one cube of the launch covers: its output block, and the run of `K` it contracts,
    /// the whole of it or a cube's share where several share a box.
    pub fn cube_block(&self) -> CubeBlock {
        let cube = self.blueprint.partitioning.cube_space().space;
        CubeBlock {
            m: cube.extent(M),
            n: cube.extent(N),
            k: cube.extent(K),
        }
    }
}

impl<Mk, SM> MatmulRoutinePrepared<Mk, SM> {
    /// The launch of `blueprint` on cubes of `cube_dim`.
    pub(crate) fn new(blueprint: MatmulBlueprint<Mk, SM>, cube_dim: CubeDim) -> Self {
        Self {
            blueprint,
            cube_dim,
        }
    }

    /// What the launch compiles.
    #[cfg(test)]
    pub(crate) fn blueprint(&self) -> &MatmulBlueprint<Mk, SM> {
        &self.blueprint
    }
}

/// Two launches compile alike where their blueprint and cube do: the cube is kept out of a
/// blueprint, and two settings preparing one blueprint can still launch different cubes.
impl<Mk, SM> PartialEq for MatmulRoutinePrepared<Mk, SM>
where
    MatmulBlueprint<Mk, SM>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.blueprint == other.blueprint && self.cube_dim == other.cube_dim
    }
}
