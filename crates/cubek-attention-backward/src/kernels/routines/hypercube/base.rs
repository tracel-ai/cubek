//! What the hypercube walks share: their decline of a device without atomic adds, their settings,
//! their blueprint, and how they place their cubes.

use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardWalk, KeyGradientElement, WalkStages, key_walk_count,
};
use cubecl::features::AtomicUsage;
use cubecl::ir::Type;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::attention::{B, KV, Q};
use cubek_microkernel::attention_backward::key_gradient::hypercube::{
    HypercubeKeyGradientMicrokernel, HypercubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel, StageRing};

/// What the hypercube routines decline on a device that cannot add `f32` atomically: the query's
/// gradient is summed across the cubes that own its keys.
pub const ATTENTION_BACKWARD_NO_ATOMIC_ADD: Decline = Decline::Unserved {
    why: "the hypercube backward pass sums the query's gradient with f32 atomic adds, which this \
          device does not have",
};

/// What a hypercube walk is measured at, its group's heads together
/// ([`HypercubeKeyGradientQuerySumRoutine`](super::HypercubeKeyGradientQuerySumRoutine)) or apart
/// ([`HypercubeGradientSumRoutine`](super::HypercubeGradientSumRoutine)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeKeyWalkSettings {
    pub microkernel: HypercubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

/// What a hypercube walk launches.
pub type HypercubeKeyWalkBlueprint =
    AttentionBackwardWalk<HypercubeKeyGradientMicrokernel, StageRing>;

/// A hypercube walk, its cubes as `cubes` places them over the microkernel's tiling, one batch
/// and key head at a time.
///
/// # Errors
///
/// [`ATTENTION_BACKWARD_NO_ATOMIC_ADD`] on a device without `f32` atomic adds, and what `cubes`,
/// the microkernel or the grid declines.
pub(super) fn key_walk(
    hw: &Hardware,
    problem: &AttentionBackwardProblem,
    settings: &HypercubeKeyWalkSettings,
    cubes: impl FnOnce(Levels) -> Result<Levels, Decline>,
) -> Result<HypercubeKeyWalkBlueprint, Decline> {
    let atomic_sum = Type::atomic(AttentionBackwardProblem::SUM);
    if !hw
        .properties
        .atomic_type_usage(atomic_sum)
        .contains(AtomicUsage::Add)
    {
        return Err(ATTENTION_BACKWARD_NO_ATOMIC_ADD);
    }
    let item = problem.work_item();
    let microkernel = HypercubeKeyGradientMicrokernel::best(hw, &item, &settings.microkernel)?;
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
        body_bytes: microkernel.share_bytes(&item),
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

/// The estimate of a hypercube walk for `problem`: its cube microkernel's walk owned by the keys
/// ([`key_walk_count`]), its gradients `written` as the routine writes them, and each cube's
/// share of the queries' gradient added into their sum
/// ([`adding_query_gradient`](crate::kernels::WalkCount::adding_query_gradient)), its planes'
/// parts met in shared memory once a stage where they split the keys and read the score through
/// windows. Its microkernel holds the work.
pub(super) fn key_walk_estimate(
    problem: &AttentionBackwardProblem,
    walk: &HypercubeKeyWalkBlueprint,
    written: KeyGradientElement,
) -> KernelEstimate {
    let (item, microkernel) = (problem.work_item(), &walk.microkernel);
    let mut count = key_walk_count(
        problem,
        &walk.partitioning,
        &microkernel.cube,
        &walk.shared_memory,
        written,
    )
    .adding_query_gradient();
    if let Some(sharing) = count.sharing.as_mut() {
        sharing.window_bytes += microkernel.share_bytes(&item) as u64;
        if microkernel.meets_in_shared() {
            sharing.drain_atomics =
                (microkernel.plane_count() * microkernel.seq_q() * item.head_dim) as u64;
        }
    }
    walk.estimate(problem, &count)
}
