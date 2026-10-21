//! The tune candidate the many-rows and few-rows tables race a float routine as.

use crate::kernels::tune::key::{CubeBlock, MatmulTuneKey};
use crate::kernels::{MatmulBlueprint, MatmulRoutineProblem};
use crate::microkernel::MatmulMicrokernel;
use cubecl::CubeDim;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::routine::Routine;

/// One candidate of a matmul table's autotune: a float routine at one of its settings,
/// prepared for the device the table is built for.
///
/// Its priority is the matmul family's one rule, the output block one cube covers against the
/// key's extents ([`MatmulShape::priority`](crate::kernels::tune::key::MatmulShape::priority)).
/// These routines snap no cut to a divisor of the problem (a cube overhanging it is masked), so
/// the prepared launch's cube box is the block a setting asks for.
#[derive(Debug, Clone)]
pub struct MatmulRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
}

impl<R: Routine + Copy> MatmulRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
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

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }
}

impl<R, M: MatmulMicrokernel, SM> MatmulRoutineCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// key's weight is packed, which a float routine does not read, or the routine declines the
    /// problem on this device.
    fn prepared(&self, key: &MatmulTuneKey) -> Option<(MatmulBlueprint<M, SM>, CubeDim)> {
        match key.is_packed() {
            true => None,
            false => self.prepared_float(key),
        }
    }

    /// The launch this candidate prepares for the float product of `key`'s extents, whatever its
    /// weight: what a packed key's product becomes once its weight is decoded.
    pub fn prepared_float(&self, key: &MatmulTuneKey) -> Option<(MatmulBlueprint<M, SM>, CubeDim)> {
        let problem = MatmulRoutineProblem::from(&key.shape());
        let (blueprint, grid) = self
            .routine
            .prepare(&self.hw, &problem, &self.settings)
            .ok()?;
        Some((blueprint, grid.launch.cube_dim))
    }

    /// Where the output block one cube of `prepared` covers places it on `key`'s extents; last
    /// where nothing is prepared, which the launch then declines before compiling anything.
    pub fn ranked(
        key: &MatmulTuneKey,
        prepared: Option<(MatmulBlueprint<M, SM>, CubeDim)>,
    ) -> CandidatePriority {
        let Some((blueprint, _)) = prepared else {
            return CandidatePriority::LAST;
        };
        let cube = blueprint.partitioning.cube_box();
        // The priority reads the output block alone; the cube's `K` stands in for the step it does
        // not read: the whole of it, or a cube's run of it where several share a box.
        key.shape().priority(CubeBlock {
            m: cube.extent(M),
            n: cube.extent(N),
            k: cube.extent(K),
        })
    }
}

impl<R, M: MatmulMicrokernel, SM: PartialEq> TuneCandidate for MatmulRoutineCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (R::Blueprint, CubeDim);

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// Where the output block one cube of the prepared launch covers places it on the key's
    /// extents; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        Self::ranked(key, self.prepared(key))
    }

    fn blueprint(&self, key: &MatmulTuneKey) -> Option<Self::Blueprint> {
        self.prepared(key)
    }
}
