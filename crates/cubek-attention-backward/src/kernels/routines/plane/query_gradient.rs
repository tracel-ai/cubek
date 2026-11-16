//! [`PlaneFragmentQueryGradientRoutine`]: the plane walk owned by the queries.

use super::PlaneFragmentAttentionBackwardSharedMemory;
use super::kernel::plane_fragment_query_gradient_kernel;
use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk, WalkStages,
};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// Every plane owns a box of queries of one head and writes their gradient and their dot product
/// of the output with its gradient, over the keys they see. The launch groups its planes in cubes
/// along the queries of one head, and every plane of a cube stages the same keys and values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentQueryGradientRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentQueryGradientRoutineSettings {
    pub microkernel: PlaneQueryGradientMicrokernelSettings,
    /// Planes the launch groups in one cube.
    pub planes: NonZeroUsize,
    /// How a plane holds its stages.
    pub ring: StageRing,
}

/// What [`PlaneFragmentQueryGradientRoutine`] launches.
pub type PlaneFragmentQueryGradientBlueprint = AttentionBackwardWalk<
    PlaneQueryGradientMicrokernel,
    PlaneFragmentAttentionBackwardSharedMemory,
>;

impl PlaneFragmentQueryGradientRoutine {
    /// A plane is as many units as the device issues its matrix instruction on, and it walks its
    /// own stages below the planes' level, so each holds its own copy. The planes take the queries
    /// of every head.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &PlaneFragmentQueryGradientRoutineSettings,
    ) -> Result<PlaneFragmentQueryGradientBlueprint, Refusal> {
        let item = problem.work_item();
        let microkernel = PlaneQueryGradientMicrokernel::best(hw, &item, &settings.microkernel)?;
        let planes = settings.planes.get();
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .tiling()
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let (grid, ring) = WalkStages {
            stage: problem.stage(S, microkernel.seq_kv()),
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
        walk: &PlaneFragmentQueryGradientBlueprint,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' gradient and dot products, and its
        // shared memory is its own.
        unsafe {
            plane_fragment_query_gradient_kernel::launch_unchecked(
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
                walk.shared_memory.ring,
                walk.shared_memory.planes,
                problem.visibility,
                [AttentionBackwardProblem::SUM, problem.element],
            );
        }
    }
}
