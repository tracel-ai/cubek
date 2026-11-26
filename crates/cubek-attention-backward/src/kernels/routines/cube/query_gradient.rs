//! [`CubeStagedQueryGradientRoutine`]: the cube staged walk owned by the queries.

use super::kernel::cube_staged_query_gradient_kernel;
use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk, WalkCount,
    WalkSharing, WalkStages,
};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Decline;
use cubecl::tile::space::ComputeScope;
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, CubeQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel, StageRing};

/// Every cube owns a box of queries of some heads of a group and writes their gradient and their
/// dot product of the output with its gradient, over the keys they see.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedQueryGradientRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedQueryGradientRoutineSettings {
    pub microkernel: CubeQueryGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

/// What [`CubeStagedQueryGradientRoutine`] launches.
pub type CubeStagedQueryGradientBlueprint =
    AttentionBackwardWalk<CubeQueryGradientMicrokernel, StageRing>;

impl CubeStagedQueryGradientRoutine {
    /// A plane is as many units as the device issues its matrix instruction on; a cube walks its
    /// stages above its planes, so they share them. The cubes take the queries of some heads of
    /// every group.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &CubeStagedQueryGradientRoutineSettings,
    ) -> Result<CubeStagedQueryGradientBlueprint, Decline> {
        let item = problem.work_item();
        let microkernel = CubeQueryGradientMicrokernel::best(hw, &item, &settings.microkernel)?;
        // Under a causal mask a later box of queries sees more keys: the heaviest start first and
        // the light ones fill in behind them.
        let cubes = microkernel.tiling().cubes(&[Q, G]);
        let cubes = match problem.visibility.is_causal() {
            true => cubes.reversed(Q),
            false => cubes,
        };
        let partitioning = Partitioning::new(problem.space(), cubes.batches(&[B, KV]).build());
        let (grid, ring) = WalkStages {
            stage: problem.stage(S, microkernel.seq_kv()),
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

    /// Its cubes' walk owned by the queries ([`WalkCount::query_walk`]), each cube staging every
    /// stage of keys and values once on its ring for all its planes, which each read it, beside
    /// their windows; its microkernel holds the work.
    pub fn estimate(
        &self,
        problem: &AttentionBackwardProblem,
        walk: &CubeStagedQueryGradientBlueprint,
    ) -> KernelEstimate {
        let microkernel = &walk.microkernel;
        let planes = microkernel.plane_count();
        let windows = planes * microkernel.plane.walk_bytes(&problem.work_item());
        let count = WalkCount::query_walk(
            problem,
            &walk.partitioning,
            ComputeScope::Cube,
            microkernel.seq_kv(),
        )
        .staged(
            &walk.shared_memory,
            WalkSharing::query_walk(planes, windows),
        );
        walk.estimate(problem, &count)
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &CubeStagedQueryGradientBlueprint,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' gradient and dot products, and each
        // cube's shared memory is its own.
        unsafe {
            cube_staged_query_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                operands.query_gradient.arg(),
                problem.seq_q as u32,
                problem.seq_kv as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                walk.shared_memory,
                problem.visibility,
                [AttentionBackwardProblem::SUM, problem.element],
            );
        }
    }
}
