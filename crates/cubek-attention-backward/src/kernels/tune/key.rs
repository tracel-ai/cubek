//! How a backward pass is told apart from another, for autotune and for the candidates it
//! measures.

use crate::kernels::AttentionBackwardProblem;
use cubecl::ir::ElemType;
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::attention::{AttentionVisibility, D, DV};
use cubek_microkernel::attention_backward::AttentionBackwardOperands;
use serde::{Deserialize, Serialize};

/// What makes two backward passes want the same routine: the head shape exact, as the model
/// fixes it, the queries, the keys and the batch anchored, where the crossovers between routines
/// live, and which keys a query sees.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct AttentionBackwardKey {
    elem: ElemType,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    val_dim: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    queries: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    keys: usize,
    #[autotune(anchor(exp(min = 1, max = 64, base = 2)))]
    batch: usize,
    visibility: AttentionVisibility,
}

/// The key `problem` is tuned under.
impl From<&AttentionBackwardProblem> for AttentionBackwardKey {
    fn from(problem: &AttentionBackwardProblem) -> Self {
        Self::new(
            problem.element,
            problem.num_kv_heads * problem.group,
            problem.num_kv_heads,
            problem.head_dim,
            problem.val_dim,
            problem.seq_q,
            problem.seq_kv,
            problem.batch,
            problem.visibility,
        )
    }
}

/// The problem a tuning key describes, at its bucket's extents: dense operands of the key's
/// element, viewed whole.
impl TryFrom<&AttentionBackwardKey> for AttentionBackwardProblem {
    type Error = Refusal;

    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_GROUP_UNEVEN`], where the key's query heads do not split evenly over
    /// its key heads.
    ///
    /// [`ATTENTION_BACKWARD_GROUP_UNEVEN`]: crate::kernels::ATTENTION_BACKWARD_GROUP_UNEVEN
    fn try_from(key: &AttentionBackwardKey) -> Result<Self, Refusal> {
        let group = Self::group_of(key.heads, key.kv_heads)?;
        // Each operand dense, its rows as long as its innermost axis.
        let row = |axis: Axis| match axis {
            D => key.head_dim,
            DV => key.val_dim,
            _ => 1,
        };
        Ok(Self {
            batch: key.batch,
            num_kv_heads: key.kv_heads,
            group,
            seq_q: key.queries,
            seq_kv: key.keys,
            head_dim: key.head_dim,
            val_dim: key.val_dim,
            visibility: key.visibility,
            element: key.elem,
            elem_alignments: AttentionBackwardOperands::from_fn(|operand| {
                let row = row(operand.innermost());
                ElemAlignment::new(&[1, row], &[row, 1])
            }),
            start_bytes: AttentionBackwardOperands::from_fn(|_| 0),
        })
    }
}
