//! The problem a routine computes.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubek_microkernel::attention::{AttentionVisibility, B, D, DV, G, KV, Q, S};
use cubek_microkernel::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardOperands, AttentionBackwardWorkItem,
};
use cubek_std::microkernel::StagedOperand;

/// What a backward pass declines before preparing anything: query heads that do not split
/// evenly over the key heads.
pub const ATTENTION_BACKWARD_GROUP_UNEVEN: Refusal = Refusal::Declined {
    why: "attention's backward pass reads a whole group of query heads per key head",
};

/// The gradients of `out[b, h, q] = softmax(query[b, h, q] · key[b, h / group]ᵀ · scale) ·
/// value[b, h / group]` against its three inputs, as the host describes it: the queries are the
/// last `seq_q` positions of the `seq_kv` keys, all of which are attended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionBackwardProblem {
    pub batch: usize,
    pub num_kv_heads: usize,
    /// Query heads per key head.
    pub group: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
    pub val_dim: usize,
    /// Which keys a query sees.
    pub visibility: AttentionVisibility,
    /// The element the inputs and the gradients are stored in.
    pub element: ElemType,
    /// The alignment of each operand's buffer in elements.
    pub elem_alignments: AttentionBackwardOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: AttentionBackwardOperands<u64>,
}

impl AttentionBackwardProblem {
    /// The element everything is summed in, and the logsumexp and the dot products stored in.
    pub(crate) const SUM: ElemType = ElemType::Float(FloatKind::F32);

    /// The query heads each key head serves: `heads` split evenly over `kv_heads`.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_GROUP_UNEVEN`], where they do not split evenly.
    pub(crate) fn group_of(heads: usize, kv_heads: usize) -> Result<usize, Refusal> {
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(ATTENTION_BACKWARD_GROUP_UNEVEN);
        }
        Ok(heads / kv_heads)
    }

    /// What every microkernel of this problem is handed.
    pub(crate) fn work_item(&self) -> AttentionBackwardWorkItem {
        AttentionBackwardWorkItem {
            element: self.element,
            sum: Self::SUM,
            group: self.group,
            head_dim: self.head_dim,
            val_dim: self.val_dim,
        }
    }

    /// Every axis of the problem at its extent: what a routine partitions.
    pub(crate) fn space(&self) -> Space {
        Space::new(&[
            (B, self.batch),
            (KV, self.num_kv_heads),
            (G, self.group),
            (Q, self.seq_q),
            (S, self.seq_kv),
            (D, self.head_dim),
            (DV, self.val_dim),
        ])
    }

    /// One stage of the operands a walk holds `extent` of along `walked`, a row of each spanning
    /// `D` and `DV`: the keys and values the query walk stages, or the queries and output
    /// gradient the key walk does.
    pub(crate) fn stage(&self, walked: Axis, extent: usize) -> [StagedOperand; 2] {
        let operand = |row: Axis, width: usize| {
            StagedOperand::new(
                self.element,
                Space::new(&[(walked, extent), (row, width)]),
                row,
            )
        };
        [operand(D, self.head_dim), operand(DV, self.val_dim)]
    }

    /// The element `operand` is stored in: the logsumexp and the dot products at the sum's, the
    /// rest at the problem's.
    pub(crate) fn elements(&self) -> AttentionBackwardOperands<ElemType> {
        AttentionBackwardOperands::from_fn(|operand| match operand {
            AttentionBackwardOperand::Logsumexp | AttentionBackwardOperand::Delta => Self::SUM,
            _ => self.element,
        })
    }
}
