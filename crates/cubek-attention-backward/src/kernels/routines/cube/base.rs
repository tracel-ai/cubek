//! What the cube staged walks owned by the keys share: their settings, their blueprint, and how
//! they place their cubes.

use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardWalk, KeyGradientElement, WalkCount, WalkSharing,
    WalkStages,
};
use cubecl::tile::launch::Decline;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::attention::{B, KV, Q};
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel, StageRing};

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
    cubes: impl FnOnce(Levels) -> Result<Levels, Decline>,
) -> Result<CubeStagedKeyWalkBlueprint, Decline> {
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

/// What a walk owned by the keys on `microkernel`'s cubes, placed by `partitioning`, moves and
/// computes: its cubes' walk ([`WalkCount::key_walk`]), each cube staging every stage of queries
/// and output's gradient once on `ring` for all its planes, which each read it and each read the
/// queries' logsumexp and dot products where they lie, beside their windows; its gradients
/// `written` as the routine writes them.
pub(crate) fn key_walk_count(
    problem: &AttentionBackwardProblem,
    partitioning: &Partitioning,
    microkernel: &CubeKeyGradientMicrokernel,
    ring: &StageRing,
    written: KeyGradientElement,
) -> WalkCount {
    let planes = microkernel.plane_count();
    let windows = planes * microkernel.plane.walk_bytes(&problem.work_item());
    let mut count = WalkCount::key_walk(
        problem,
        partitioning,
        ComputeScope::Cube,
        microkernel.seq_q(),
        written,
    )
    .staged(ring, WalkSharing::key_walk(planes, windows));
    for read in &mut count.reads {
        if matches!(
            read.operand,
            AttentionBackwardOperand::Logsumexp | AttentionBackwardOperand::Delta
        ) {
            read.movers = planes as u64;
        }
    }
    count
}

/// The estimate of a cube staged walk owned by the keys for `problem` ([`key_walk_count`]); its
/// microkernel holds the work.
pub(super) fn key_walk_estimate(
    problem: &AttentionBackwardProblem,
    walk: &CubeStagedKeyWalkBlueprint,
    written: KeyGradientElement,
) -> KernelEstimate {
    let count = key_walk_count(
        problem,
        &walk.partitioning,
        &walk.microkernel,
        &walk.shared_memory,
        written,
    );
    walk.estimate(problem, &count)
}
