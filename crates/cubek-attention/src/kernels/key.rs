//! How an attention is told apart from another, for a caller that measures the
//! routines and for the candidates it measures.
//!
//! A key describes the *problem* — the head shape, the extents, which keys a
//! query sees and the scale its scores take — so it lives beside the plans
//! rather than beside the autotune that consumes it. A caller builds the autotune; it is
//! this crate that knows what makes two attentions the same attention.
//!
//! One key for a prompt, a chunk of one and a decode step: the microkernel
//! family serves all three, so one key states the queries and the keys an autotune
//! is asked about, and each routine reads the problem it serves off it,
//! [`crate::kernels::AttentionRoutineProblem`]. It reads the key alone,
//! because the key is the whole of what cubecl hands a priority function.

use crate::kernels::{
    ATTENTION_GROUP_UNEVEN, AttentionCacheLayout, AttentionRoutineProblem, AttentionVisibility,
    QueryDimension, ScoreScale,
};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKeyable};
use cubek_microkernel::attention::AttentionOperands;
use serde::{Deserialize, Serialize};

/// The elements an [`AttentionKey`]'s operands are read in: the query's, which the output is
/// written in, and the keys' and the values' each its own.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AttentionKeyElements {
    pub query: ElemType,
    pub key: ElemType,
    pub value: ElemType,
}

impl AttentionKeyElements {
    /// Every operand in `elem`.
    pub fn one(elem: ElemType) -> Self {
        Self {
            query: elem,
            key: elem,
            value: elem,
        }
    }
}

/// What makes two attentions want the same kernel.
///
/// The head shape is **exact** — it is fixed by the model config, and the GQA
/// ratio decides how much expansion a flash candidate pays. The queries, the
/// keys and the batch are bucketed: they are the dimensions a prompt and a cache
/// vary freely, and where the crossovers between kernels live.
///
/// **A bucket's representative is the extent a call is rounded up to**, and a
/// candidate's duplicate proof is taken there rather than at the call in hand, so
/// the candidates that collapse at some length inside a bucket collapse together. The round
/// itself runs on the call in hand.
///
/// **The queries and the keys are bucketed in powers of two, whatever the autotune level**
/// ([`positions_bucket`](Self::positions_bucket)). A caller buckets its depth in powers of two and
/// prefills in chunks of a power of two, so the bucket's extents are the ones its calls run.
/// cubecl's anchor rounds in powers of 1.5 at the extensive level: a chunk of 512 queries would be
/// keyed with calls of 657, a second box of 145 queries it does not run.
///
/// **The keys are the depth the caller states, never a cache's capacity.** How many positions
/// each sequence attends lives on device, so that every call of one bucket replays the same
/// launch, the graph-capture precondition; the caller states the call's depth on the host, at its
/// own bucket ([`AttentionTuneInputs::new`]), and the key anchors it. A prompt into an empty cache
/// and a chunk deep into a long one are keyed apart, and a cache's buffers may be any size.
///
/// [`AttentionTuneInputs::new`]: crate::kernels::AttentionTuneInputs::new
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKeyable)]
pub struct AttentionKey {
    /// The element each operand is read in: a cache's keys and values may each be stored in
    /// another than the query's, which a routine reads differently, so a winner measured over one
    /// cache does not serve another.
    elements: AttentionKeyElements,
    /// Query heads.
    heads: usize,
    /// Key/value heads (GQA when fewer than `heads`).
    kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Query positions, at their bucket ([`positions_bucket`](Self::positions_bucket)).
    queries: usize,
    /// Key positions: the depth the call attends, as its caller states it, at its bucket
    /// ([`positions_bucket`](Self::positions_bucket)).
    keys: usize,
    #[autotune(anchor(exp(min = 1, max = 64, base = 2)))]
    batch: usize,
    /// Which keys a query sees: a causal mask and a window are part of the
    /// computation, not a detail of it, and the flash routines ([`forward`](crate::forward)) serve
    /// neither a window nor a chunk's alignment.
    visibility: AttentionVisibility,
    /// Which scale the scores take: the flash routines apply
    /// `1 / sqrt(head_dim)` and no other, so a winner measured at one cannot
    /// serve the other.
    scale: ScoreScale,
    /// Positions a page holds where the keys and values are read through a page table: a paged
    /// read costs what a dense one does not, so a winner measured on one does not serve the other.
    page_size: Option<usize>,
    /// Whether a mask tensor narrows the keys each query sees: its read is a cost per score
    /// the unmasked launch does not pay.
    masked: bool,
    /// Whether every sequence's queries lie end to end, each read through its run: the routines
    /// that merge split keys outside the tiles decline them, so a winner measured on one does not
    /// serve the other.
    packed_queries: bool,
}

impl AttentionKey {
    /// The extent a key holds for a call of `positions` query or key positions: the power of two
    /// at or above them.
    pub fn positions_bucket(positions: usize) -> usize {
        positions.next_power_of_two()
    }

    /// Query positions, at the bucket's own extent.
    pub fn seq_q(&self) -> usize {
        self.queries
    }

    /// Key positions, at the bucket's own extent.
    pub fn seq_kv(&self) -> usize {
        self.keys
    }

    /// The element each operand binds at.
    pub fn elements(&self) -> AttentionKeyElements {
        self.elements
    }

    /// Key/value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Query heads per key head, where they divide.
    pub fn group(&self) -> Option<usize> {
        (self.kv_heads > 0 && self.heads.is_multiple_of(self.kv_heads))
            .then_some(self.heads / self.kv_heads)
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Batch, at the bucket's own extent.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Which keys a query sees.
    pub fn visibility(&self) -> AttentionVisibility {
        self.visibility
    }

    /// Which scale the scores take.
    pub fn scale(&self) -> ScoreScale {
        self.scale
    }

    /// Positions a page holds, where the keys and values are read through a page table.
    pub fn page_size(&self) -> Option<usize> {
        self.page_size
    }

    /// Whether every sequence's queries lie end to end.
    pub fn packed_queries(&self) -> bool {
        self.packed_queries
    }
}

/// The problem a tuning **key** describes, at its bucket's extents ([`AttentionKey`]): operands of
/// the key's elements, the output at the query's, the query `[b, h, q, d]`, the keys and values
/// `[b, kv, s, d]`, read through a page table where the key has a page size, and the output `[b, q,
/// h, d]`, the value dim the head dim. A key carries no value dim of its own: every attention
/// reaching autotune has `DV = D`.
impl TryFrom<&AttentionKey> for AttentionRoutineProblem {
    type Error = Decline;

    /// # Errors
    ///
    /// [`ATTENTION_GROUP_UNEVEN`], where the key's query heads do not split evenly over its key
    /// heads.
    fn try_from(key: &AttentionKey) -> Result<Self, Decline> {
        let group = key.group().ok_or(ATTENTION_GROUP_UNEVEN)?;
        let (elements, head) = (key.elements(), key.head_dim());
        let rows = |innermost: usize| ElemAlignment::new(&[1, innermost], &[innermost, 1]);
        let (group, queries, visibility) = QueryDimension::new(key.seq_q(), key.visibility()).read(
            group,
            key.seq_q(),
            key.visibility(),
        );
        Ok(Self {
            batch: key.batch(),
            num_kv_heads: key.num_kv_heads(),
            group,
            seq_q: queries,
            seq_kv: key.seq_kv(),
            head_dim: head,
            val_dim: head,
            visibility,
            elements: AttentionOperands {
                query: elements.query,
                key: elements.key,
                value: elements.value,
                out: elements.query,
            },
            elem_alignments: AttentionOperands {
                query: rows(head),
                key: rows(head),
                value: rows(head),
                out: rows(head),
            },
            // A key describes operands a launch allocates or views whole.
            start_bytes: AttentionOperands {
                query: 0,
                key: 0,
                value: 0,
                out: 0,
            },
            cache: match key.page_size() {
                Some(page_size) => AttentionCacheLayout::Pages { page_size },
                None => AttentionCacheLayout::Sequences,
            },
            packed_queries: key.packed_queries(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A power of two is its own bucket, so a chunk of 512 queries is keyed at 512, and any
    /// extent past it at the next.
    #[test]
    fn a_bucket_is_the_power_of_two_at_or_above() {
        let buckets = [1, 300, 512, 513, 2048, 2049].map(AttentionKey::positions_bucket);
        assert_eq!(buckets, [1, 512, 512, 1024, 2048, 4096]);
    }
}
