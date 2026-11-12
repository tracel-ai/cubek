//! [`UnitKeyGradientRoutine`]: the unit walk owned by the keys.

use super::kernel::unit_key_gradient_kernel;
use super::unit_cube;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::{B, KV, S};
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, UnitKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::Grid;

/// Every unit owns keys of one key head, taken in turn and on into the next head, and writes their
/// gradient and their values', over every query of the group's heads that sees them. It reads the
/// queries' dot products of the output with its gradient, which a walk owned by the queries wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitKeyGradientRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitKeyGradientRoutineSettings {
    pub microkernel: UnitKeyGradientMicrokernelSettings,
    /// Planes the launch groups its units in, a cube.
    pub planes: NonZeroUsize,
}

/// What [`UnitKeyGradientRoutine`] launches.
pub type UnitKeyGradientBlueprint = AttentionBackwardWalk<UnitKeyGradientMicrokernel>;

impl UnitKeyGradientRoutine {
    /// The units take the keys of every key head.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &UnitKeyGradientRoutineSettings,
    ) -> Result<UnitKeyGradientBlueprint, Refusal> {
        let microkernel =
            UnitKeyGradientMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let (cube, resources) = unit_cube(hw, settings.planes);
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel.tiling().launch_units(&[S, B, KV], cube).build(),
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

    pub fn dispatch(&self, tensors: &AttentionBackwardTensors, walk: &UnitKeyGradientBlueprint) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each unit writes only its own keys' and values' gradients.
        unsafe {
            unit_key_gradient_kernel::launch_unchecked(
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
                problem.visibility,
                AttentionBackwardProblem::SUM,
            );
        }
    }
}
