//! The problem a routine computes.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::attention::{AttentionVisibility, B, D, DV, G, KV, Q, S};
use cubek_microkernel::attention_backward::{AttentionBackwardOperands, AttentionBackwardWorkItem};

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
    pub const SUM: ElemType = ElemType::Float(FloatKind::F32);

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> AttentionBackwardWorkItem {
        AttentionBackwardWorkItem {
            element: self.element,
            sum: Self::SUM,
            group: self.group,
            head_dim: self.head_dim,
            val_dim: self.val_dim,
        }
    }

    /// Every axis of the problem at its extent: what a routine partitions.
    pub fn space(&self) -> Space {
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

    /// The element `operand` is stored in: the logsumexp and the dot products at the sum's, the
    /// rest at the problem's.
    pub fn elements(&self) -> AttentionBackwardOperands<ElemType> {
        AttentionBackwardOperands {
            query: self.element,
            key: self.element,
            value: self.element,
            out: self.element,
            out_gradient: self.element,
            logsumexp: Self::SUM,
            delta: Self::SUM,
            query_gradient: self.element,
            key_gradient: self.element,
            value_gradient: self.element,
        }
    }
}
