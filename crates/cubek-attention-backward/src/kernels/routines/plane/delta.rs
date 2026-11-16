//! [`PlaneDeltaRoutine`]: each query's dot product of the output with its gradient, alone.

use super::kernel::plane_delta_kernel;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::attention::{B, G, KV, Q};
use cubek_microkernel::attention_backward::delta::plane::{
    PlaneDeltaMicrokernel, PlaneDeltaMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// Every plane owns a box of queries of one head, a plane a cube, and writes their dot product of
/// the output with its gradient: what a walk owned by the keys reads where no walk owned by the
/// queries ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneDeltaRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneDeltaRoutineSettings {
    pub microkernel: PlaneDeltaMicrokernelSettings,
}

/// What [`PlaneDeltaRoutine`] launches.
pub type PlaneDeltaBlueprint = AttentionBackwardWalk<PlaneDeltaMicrokernel>;

impl PlaneDeltaRoutine {
    /// The planes take the queries of every head, a plane a cube.
    ///
    /// # Errors
    ///
    /// A device that commits to no plane width, and what the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &PlaneDeltaRoutineSettings,
    ) -> Result<PlaneDeltaBlueprint, Refusal> {
        let microkernel =
            PlaneDeltaMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .tiling()
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, 1)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let grid = Grid::new(
            hw,
            &partitioning,
            CubeResources {
                plane_width: hw.plane_dim()?,
                shared_memory: 0,
            },
        )?;
        Ok(AttentionBackwardWalk::new(
            hw,
            problem,
            microkernel,
            partitioning,
            grid,
            (),
        ))
    }

    pub fn dispatch(&self, tensors: &AttentionBackwardTensors, walk: &PlaneDeltaBlueprint) {
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' dot products.
        unsafe {
            plane_delta_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.out.arg(),
                operands.out_gradient.arg(),
                operands.delta.arg(),
                binding.launcher.partitioning_arg(),
                AttentionBackwardProblem::SUM,
            );
        }
    }
}
