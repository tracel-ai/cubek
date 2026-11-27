//! [`HypercubeKeyGradientQuerySumRoutine`]: the hypercube walk, every head of a group in one cube:
//! the keys' gradients, and the queries' gradient as a sum.

use super::kernel::hypercube_key_gradient_query_sum_kernel;
use super::{HypercubeKeyWalkBlueprint, HypercubeKeyWalkSettings, key_walk, key_walk_estimate};
use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardTensors, KeyGradientElement, QueryGradientSum,
};
use cubecl::tile::launch::Decline;
use cubek_microkernel::attention::S;
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;

/// Every cube owns a box of keys of one key head, walks every head of its group in turn, and
/// writes the keys' gradient and the values'. It adds its share of the queries' gradient into the
/// [`QueryGradientSum`], which a cast routine casts once every cube is done. It reads the queries'
/// dot products of the output with its gradient, which the delta routine wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeKeyGradientQuerySumRoutine;

impl HypercubeKeyGradientQuerySumRoutine {
    /// The cubes take the keys of every key head.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_NO_ATOMIC_ADD`](super::ATTENTION_BACKWARD_NO_ATOMIC_ADD) on a device
    /// without `f32` atomic adds, and what the microkernel or the grid declines.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &HypercubeKeyWalkSettings,
    ) -> Result<HypercubeKeyWalkBlueprint, Decline> {
        key_walk(hw, problem, settings, |tiling| Ok(tiling.cubes(&[S])))
    }

    /// Its cubes' walk owned by the keys, writing the keys' and values' gradients and adding
    /// into the queries' sum ([`key_walk_estimate`]).
    pub fn estimate(
        &self,
        problem: &AttentionBackwardProblem,
        walk: &HypercubeKeyWalkBlueprint,
    ) -> KernelEstimate {
        key_walk_estimate(problem, walk, KeyGradientElement::Stored)
    }

    /// The cubes of `walk` writing `tensors`' keys' and values' gradients, and adding their shares
    /// of the queries' into `query_sum`.
    ///
    /// # Guarantees
    ///
    /// As [`UnitQueryGradientRoutine::dispatch`](crate::kernels::UnitQueryGradientRoutine::dispatch).
    ///
    /// # Errors
    ///
    /// As [`AttentionBackwardLaunch::dispatch`](crate::kernels::AttentionBackwardLaunch::dispatch).
    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &HypercubeKeyWalkBlueprint,
        query_sum: &QueryGradientSum,
    ) -> Result<(), Decline> {
        let problem = tensors.problem();
        let binding = tensors.bind(walk)?;
        let query_summed = tensors.bind_beside(
            &binding,
            walk,
            &query_sum.0,
            AttentionBackwardOperand::QueryGradient,
        )?;
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own keys' and values' gradients, and only adds
        // into the query's gradient's sum, atomically; each cube's shared memory is its own.
        unsafe {
            hypercube_key_gradient_query_sum_kernel::launch_unchecked(
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
                query_summed.atomic::<f32>(),
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
        Ok(())
    }
}
