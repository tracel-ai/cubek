//! How a backward pass is told apart from another, for autotune and for the candidates it
//! measures.

use crate::{ATTENTION_BACKWARD_GROUP_UNEVEN, AttentionBackwardProblem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::attention::AttentionVisibility;
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

impl AttentionBackwardKey {
    /// The key `problem` is tuned under.
    pub fn of(problem: &AttentionBackwardProblem) -> Self {
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
    fn try_from(key: &AttentionBackwardKey) -> Result<Self, Refusal> {
        if key.kv_heads == 0 || !key.heads.is_multiple_of(key.kv_heads) {
            return Err(ATTENTION_BACKWARD_GROUP_UNEVEN);
        }
        let rows = |innermost: usize| ElemAlignment::new(&[1, innermost], &[innermost, 1]);
        let (head, value) = (rows(key.head_dim), rows(key.val_dim));
        Ok(Self {
            batch: key.batch,
            num_kv_heads: key.kv_heads,
            group: key.heads / key.kv_heads,
            seq_q: key.queries,
            seq_kv: key.keys,
            head_dim: key.head_dim,
            val_dim: key.val_dim,
            visibility: key.visibility,
            element: key.elem,
            elem_alignments: AttentionBackwardOperands {
                query: head,
                key: head,
                value,
                out: value,
                out_gradient: value,
                logsumexp: rows(1),
                delta: rows(1),
                query_gradient: head,
                key_gradient: head,
                value_gradient: value,
            },
            start_bytes: AttentionBackwardOperands {
                query: 0,
                key: 0,
                value: 0,
                out: 0,
                out_gradient: 0,
                logsumexp: 0,
                delta: 0,
                query_gradient: 0,
                key_gradient: 0,
                value_gradient: 0,
            },
        })
    }
}
