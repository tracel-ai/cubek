//! [`PlaneFragmentKeyGradientRoutine`]: the plane walk owned by the keys.

use super::PlaneFragmentAttentionBackwardSharedMemory;
use super::kernel::plane_fragment_key_gradient_kernel;
use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk, WalkStages,
};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::attention::{B, KV, Q, S};
use cubek_microkernel::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, PlaneKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// Every plane owns a box of keys of one key head and writes their gradient and their values',
/// over every query of the group's heads that sees them. It reads the queries' dot products of
/// the output with its gradient, which a walk owned by the queries wrote. The launch groups its
/// planes in cubes along the keys of one key head, and every plane of a cube stages the same
/// queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentKeyGradientRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentKeyGradientRoutineSettings {
    pub microkernel: PlaneKeyGradientMicrokernelSettings,
    /// Planes the launch groups in one cube.
    pub planes: NonZeroUsize,
    /// How a plane holds its stages.
    pub ring: StageRing,
}

/// What [`PlaneFragmentKeyGradientRoutine`] launches.
pub type PlaneFragmentKeyGradientBlueprint =
    AttentionBackwardWalk<PlaneKeyGradientMicrokernel, PlaneFragmentAttentionBackwardSharedMemory>;

impl PlaneFragmentKeyGradientRoutine {
    /// As [`PlaneFragmentQueryGradientRoutine`](super::PlaneFragmentQueryGradientRoutine)'s
    /// planes, over the keys of every key head.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &PlaneFragmentKeyGradientRoutineSettings,
    ) -> Result<PlaneFragmentKeyGradientBlueprint, Refusal> {
        let item = problem.work_item();
        let microkernel = PlaneKeyGradientMicrokernel::best(hw, &item, &settings.microkernel)?;
        let planes = settings.planes.get();
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .tiling()
                .launch_planes(
                    &[S, B, KV],
                    PlaneOrder::new(&[(S, planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let (grid, ring) = WalkStages {
            stage: problem.stage(Q, microkernel.seq_q()),
            ring: settings.ring,
            filling_planes: 1,
            copies: planes,
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
            PlaneFragmentAttentionBackwardSharedMemory { ring, planes },
        ))
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &PlaneFragmentKeyGradientBlueprint,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own keys' and values' gradients, and its shared
        // memory is its own.
        unsafe {
            plane_fragment_key_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                operands.key_gradient.arg(),
                operands.value_gradient.arg(),
                problem.seq_q as u32,
                problem.seq_kv as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                walk.shared_memory.ring,
                walk.shared_memory.planes,
                problem.visibility,
                [AttentionBackwardProblem::SUM, problem.element],
            );
        }
    }
}
