//! The cube staged routines: every cube owns a box of queries of some heads of a group, or of
//! keys of one key head, its planes each on a matrix instruction. A cube stages what its walk
//! holds still once, and each stage of what it walks once, for all its planes, met on
//! `sync_cube`; each plane holds its own windows.
//!
//! What the plane routines' planes each read again, the cube reads once: the keys and values every
//! head and every plane along the queries read in the walk owned by the queries, and the queries
//! every plane along the keys reads in the walk owned by the keys.

mod kernel;
mod key_gradient;
mod key_gradient_sum;
mod query_gradient;

pub use key_gradient::*;
pub use key_gradient_sum::*;
pub use query_gradient::*;

use crate::kernels::{AttentionBackwardProblem, AttentionBackwardWalk, WalkStages};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::attention::{B, KV, Q};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedKeyGradientRoutineSettings {
    pub microkernel: CubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

/// What a cube staged walk owned by the keys launches.
pub type CubeStagedKeyGradientBlueprint =
    AttentionBackwardWalk<CubeKeyGradientMicrokernel, StageRing>;

/// A cube staged walk owned by the keys, its cubes as `cubes` places them over the microkernel's
/// tiling, one batch and key head at a time.
fn key_walk(
    hw: &Hardware,
    problem: &AttentionBackwardProblem,
    settings: &CubeStagedKeyGradientRoutineSettings,
    cubes: impl FnOnce(Levels) -> Result<Levels, Refusal>,
) -> Result<CubeStagedKeyGradientBlueprint, Refusal> {
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
