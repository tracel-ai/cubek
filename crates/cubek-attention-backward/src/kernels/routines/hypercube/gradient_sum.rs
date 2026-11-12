//! [`HypercubeGradientSumRoutine`]: the hypercube walk, the heads of a group apart: every gradient
//! as a sum.

use super::kernel::hypercube_gradient_sum_kernel;
use super::{HypercubeKeyWalkBlueprint, HypercubeKeyWalkSettings, key_walk};
use crate::kernels::routines::heads_apart::cubes_with_heads_apart;
use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardTensors, KeyGradientSums, QueryGradientSum,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_std::Hardware;

/// [`HypercubeKeyGradientQuerySumRoutine`](super::HypercubeKeyGradientQuerySumRoutine) with the heads of a group
/// split over cubes: every cube walks one head and adds its part of the keys' and values'
/// gradients into the [`KeyGradientSums`] too, which every cube of the group adds into. It writes
/// no gradient: cast routines do, once every cube is done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeGradientSumRoutine;

impl HypercubeGradientSumRoutine {
    /// The cubes take the keys of every key head and each head of its group.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_NO_ATOMIC_ADD`](super::ATTENTION_BACKWARD_NO_ATOMIC_ADD) on a device
    /// without `f32` atomic adds, a group of one head, which has no heads to set apart, and what
    /// the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &HypercubeKeyWalkSettings,
    ) -> Result<HypercubeKeyWalkBlueprint, Refusal> {
        key_walk(hw, problem, settings, |tiling| {
            cubes_with_heads_apart(tiling, problem.group)
        })
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &HypercubeKeyWalkBlueprint,
        key_sums: &KeyGradientSums,
        query_sum: &QueryGradientSum,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(walk);
        let bind_sum = |tensor, operand| tensors.bind_beside(&binding, walk, tensor, operand);
        let key_summed = bind_sum(&key_sums.key, AttentionBackwardOperand::KeyGradient);
        let value_summed = bind_sum(&key_sums.value, AttentionBackwardOperand::ValueGradient);
        let query_summed = bind_sum(&query_sum.0, AttentionBackwardOperand::QueryGradient);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane only adds into its own keys' and values' sums and the query's
        // gradient's sum, atomically; each cube's shared memory is its own.
        unsafe {
            hypercube_gradient_sum_kernel::launch_unchecked(
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
                query_summed.atomic::<f32>(),
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
