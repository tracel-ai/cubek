//! [`CastQueryGradientRoutine`], and the sum it reads.

use super::kernel::cast_query_gradient_kernel;
use crate::kernels::{AttentionBackwardProblem, AttentionBackwardTensors, AttentionBackwardWalk};
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_std::Hardware;

/// The queries' gradient, summed in `f32` by every cube that owns keys those queries see: what
/// the hypercube walks add into.
#[derive(Debug, Clone)]
pub struct QueryGradientSum(pub(crate) PlainTensor);

impl QueryGradientSum {
    /// The sum at zero, as every cube adds into it.
    pub(crate) fn zeros(tensors: &AttentionBackwardTensors) -> Self {
        let problem = tensors.problem();
        Self(PlainTensor::of_handle(TensorHandle::zeros(
            tensors.client(),
            [
                problem.batch,
                problem.num_kv_heads * problem.group,
                problem.seq_q,
                problem.head_dim,
            ],
            AttentionBackwardProblem::SUM,
        )))
    }
}

/// Every plane casts its queries' gradient out of the [`QueryGradientSum`]. It runs under the
/// planes of a walk owned by the queries, the delta routine's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CastQueryGradientRoutine;

/// What [`CastQueryGradientRoutine`] launches: the placement of the walk it runs under.
pub type CastQueryGradientBlueprint = AttentionBackwardWalk<()>;

impl CastQueryGradientRoutine {
    /// The cast under `queries`' planes, the gradient and the sum bound at one width, which the
    /// sum's element must allow too.
    pub fn prepare<M, SM>(
        &self,
        hw: &Hardware,
        queries: &AttentionBackwardWalk<M, SM>,
    ) -> CastQueryGradientBlueprint {
        let mut walk = queries.placement();
        let sum_ceiling = hw.vector_ceiling(AttentionBackwardProblem::SUM);
        walk.widths = walk.widths.map(|operand, &width| match operand {
            AttentionBackwardOperand::QueryGradient => width.min(sum_ceiling),
            _ => width,
        });
        walk
    }

    pub fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        walk: &CastQueryGradientBlueprint,
        sum: &QueryGradientSum,
    ) {
        let binding = tensors.bind(walk);
        let sum = tensors.bind_beside(
            &binding,
            walk,
            &sum.0,
            AttentionBackwardOperand::QueryGradient,
        );

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' gradient, out of the sum the
        // launches before finished adding into, which the launches' order sees done.
        unsafe {
            cast_query_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                sum.arg(),
                binding.operands.query_gradient.arg(),
                binding.launcher.partitioning_arg(),
            );
        }
    }
}
