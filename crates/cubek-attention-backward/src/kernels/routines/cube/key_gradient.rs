//! [`CubeStagedKeyGradientRoutine`]: the cube staged walk owned by the keys.

use super::kernel::cube_staged_key_gradient_kernel;
use super::{CubeStagedKeyGradientBlueprint, CubeStagedKeyGradientRoutineSettings, key_walk};
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::S;
use cubek_std::Hardware;

/// Every cube owns a box of keys of one key head and writes their gradient and their values',
/// over every query of every head of the group that sees them. It reads the queries' dot products
/// of the output with its gradient, which a walk owned by the queries wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedKeyGradientRoutine;

impl CubeStagedKeyGradientRoutine {
    /// The cubes take the keys of every key head, each walking every head of its group in turn.
    ///
    /// # Errors
    ///
    /// What the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &CubeStagedKeyGradientRoutineSettings,
    ) -> Result<CubeStagedKeyGradientBlueprint, Refusal> {
        key_walk(hw, problem, settings, |tiling| Ok(tiling.cubes(&[S])))
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &CubeStagedKeyGradientBlueprint,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own keys' and values' gradients, and each cube's
        // shared memory is its own.
        unsafe {
            cube_staged_key_gradient_kernel::launch_unchecked(
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
                walk.shared_memory,
                problem.visibility,
                [AttentionBackwardProblem::SUM, problem.element],
            );
        }
    }
}
