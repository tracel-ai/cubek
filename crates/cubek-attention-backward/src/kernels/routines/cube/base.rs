//! What the cube staged walks owned by the keys share: their settings, their blueprint, and how
//! they place their cubes.

use crate::kernels::{AttentionBackwardProblem, AttentionBackwardWalk, WalkStages};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::attention::{B, KV, Q};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// What a cube staged walk owned by the keys is measured at, its group's heads together
/// ([`CubeStagedKeyGradientRoutine`](super::CubeStagedKeyGradientRoutine)) or apart
/// ([`CubeStagedKeyGradientSumRoutine`](super::CubeStagedKeyGradientSumRoutine)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedKeyWalkSettings {
    pub microkernel: CubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

/// What a cube staged walk owned by the keys launches.
pub type CubeStagedKeyWalkBlueprint = AttentionBackwardWalk<CubeKeyGradientMicrokernel, StageRing>;

/// A cube staged walk owned by the keys, its cubes as `cubes` places them over the microkernel's
/// tiling, one batch and key head at a time.
pub(super) fn key_walk(
    hw: &Hardware,
    problem: &AttentionBackwardProblem,
    settings: &CubeStagedKeyWalkSettings,
    cubes: impl FnOnce(Levels) -> Result<Levels, Refusal>,
) -> Result<CubeStagedKeyWalkBlueprint, Refusal> {
    let item = problem.work_item();
    let microkernel = CubeKeyGradientMicrokernel::best(hw, &item, &settings.microkernel)?;
    let partitioning = Partitioning::new(
        problem.space(),
        cubes(microkernel.tiling())?.batches(&[B, KV]).build(),
    );
    let (grid, ring) = WalkStages {
        stage: problem.stage(Q, microkernel.seq_q()),
        ring: settings.ring,
        filling_planes: microkernel.plane_count(),
        copies: 1,
        window_bytes: microkernel.window_bytes(&item),
        body_bytes: 0,
    }
    .grid(hw, &partitioning)?;
    Ok(AttentionBackwardWalk::new(
        hw,
        problem,
        microkernel,
        partitioning,
        grid,
        ring,
    ))
}
