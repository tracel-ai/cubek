//! [`CastKeyGradientRoutine`], and the sums it reads.

use super::kernel::cast_key_gradient_kernel;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk};
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention_backward::AttentionBackwardOperand;

/// The keys' and the values' gradients, summed in `f32` by every cube of a group, each holding
/// fewer heads than the group's: what a walk owned by the keys with its heads apart adds into.
#[derive(Debug, Clone)]
pub struct KeyGradientSums {
    pub(crate) key: PlainTensor,
    pub(crate) value: PlainTensor,
}

impl KeyGradientSums {
    /// Both sums at zero, as every cube of a group adds into them.
    pub(crate) fn zeros(tensors: &AttentionBackwardTensors) -> Self {
        let problem = tensors.problem();
        let sum = |width: usize| {
            PlainTensor::of_handle(TensorHandle::zeros(
                tensors.client(),
                [problem.batch, problem.num_kv_heads, problem.seq_kv, width],
                AttentionBackwardProblem::SUM,
            ))
        };
        Self {
            key: sum(problem.head_dim),
            value: sum(problem.val_dim),
        }
    }
}

/// Every plane of the cubes on a group's first head casts its keys' and values' gradients out of
/// the [`KeyGradientSums`]: the other cubes of the group cover the same keys. It runs under the
/// planes of the walk that added into the sums.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CastKeyGradientRoutine;

/// What [`CastKeyGradientRoutine`] launches: the summing walk's placement.
pub type CastKeyGradientBlueprint = AttentionBackwardWalk<()>;

impl CastKeyGradientRoutine {
    /// The cast under `summed`'s planes, at its widths.
    pub fn prepare<M, SM>(
        &self,
        summed: &AttentionBackwardWalk<M, SM>,
    ) -> CastKeyGradientBlueprint {
        summed.placement()
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &CastKeyGradientBlueprint,
        sums: &KeyGradientSums,
    ) {
        let binding = tensors.bind(walk);
        let (key_sum, value_sum) = (
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
        // its tiles. Each plane of the cubes on a group's first head writes only its own keys'
        // and values' gradients, out of sums the launch before finished adding into, which the
        // launches' order sees done.
        unsafe {
            cast_key_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                key_sum.arg(),
                value_sum.arg(),
                operands.key_gradient.arg(),
                operands.value_gradient.arg(),
                binding.launcher.partitioning_arg(),
            );
        }
    }
}
