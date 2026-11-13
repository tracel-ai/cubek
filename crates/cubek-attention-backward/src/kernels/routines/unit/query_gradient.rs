//! [`UnitQueryGradientRoutine`]: the unit walk owned by the queries.

use super::kernel::unit_query_gradient_kernel;
use super::unit_cube;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::{B, G, KV, Q};
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, UnitQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::Grid;

/// Every unit owns queries of one head, taken in turn and on into the next head, and writes their
/// gradient and their dot product of the output with its gradient, over the keys they see.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitQueryGradientRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitQueryGradientRoutineSettings {
    pub microkernel: UnitQueryGradientMicrokernelSettings,
    /// Planes the launch groups its units in, a cube.
    pub planes: NonZeroUsize,
}

/// What [`UnitQueryGradientRoutine`] launches.
pub type UnitQueryGradientBlueprint = AttentionBackwardWalk<UnitQueryGradientMicrokernel>;

impl UnitQueryGradientRoutine {
    /// The units take the queries of every head.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &UnitQueryGradientRoutineSettings,
    ) -> Result<UnitQueryGradientBlueprint, Refusal> {
        let microkernel =
            UnitQueryGradientMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let (cube, resources) = unit_cube(hw, settings.planes);
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .tiling()
                .launch_units(&[Q, G, B, KV], cube)
                .build(),
        );
        let grid = Grid::new(hw, &partitioning, resources)?;
        Ok(AttentionBackwardWalk::new(
            hw,
            problem,
            microkernel,
            partitioning,
            grid,
            (),
        ))
    }

    pub fn dispatch(&self, tensors: &AttentionBackwardTensors, walk: &UnitQueryGradientBlueprint) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each unit writes only its own queries' gradient and dot products.
        unsafe {
            unit_query_gradient_kernel::launch_unchecked(
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
                problem.visibility,
                AttentionBackwardProblem::SUM,
            );
        }
    }
}
