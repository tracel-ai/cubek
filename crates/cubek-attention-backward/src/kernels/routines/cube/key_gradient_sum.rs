//! [`CubeStagedKeyGradientSumRoutine`]: the cube staged walk owned by the keys, the heads of a
//! group apart.

use super::kernel::cube_staged_key_gradient_sum_kernel;
use super::{CubeStagedKeyWalkBlueprint, CubeStagedKeyWalkSettings, key_walk};
use crate::kernels::routines::heads_apart::cubes_with_heads_apart;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, KeyGradientSums};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_std::Hardware;

/// [`CubeStagedKeyGradientRoutine`](super::CubeStagedKeyGradientRoutine) with the heads of a
/// group split over cubes: every cube walks one head and adds its part of the keys' and values'
/// gradients into the [`KeyGradientSums`], which every cube of the group adds into. It writes no
/// gradient: a cast routine does, once every cube is done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedKeyGradientSumRoutine;

impl CubeStagedKeyGradientSumRoutine {
    /// The cubes take the keys of every key head and each head of its group.
    ///
    /// # Errors
    ///
    /// A group of one head, which has no heads to set apart, and what the microkernel or the grid
    /// declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &CubeStagedKeyWalkSettings,
    ) -> Result<CubeStagedKeyWalkBlueprint, Refusal> {
        key_walk(hw, problem, settings, |tiling| {
            cubes_with_heads_apart(tiling, problem.group)
        })
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &CubeStagedKeyWalkBlueprint,
        sums: &KeyGradientSums,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let (key_summed, value_summed) = (
            tensors.bind_beside(
                &binding,
                walk,
                &sums.key,
                AttentionBackwardOperand::KeyGradient,
            ),
            tensors.bind_beside(
                &binding,
                walk,
                &sums.value,
                AttentionBackwardOperand::ValueGradient,
            ),
        );
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane only adds into its own keys' and values' sums, atomically, and
        // each cube's shared memory is its own.
        unsafe {
            cube_staged_key_gradient_sum_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                key_summed.atomic::<f32>(),
                value_summed.atomic::<f32>(),
                problem.seq_q as u32,
                problem.seq_kv as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                walk.shared_memory,
                problem.visibility,
                problem.element,
            );
        }
    }
}
