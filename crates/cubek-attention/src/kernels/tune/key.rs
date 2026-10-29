//! How an attention is told apart from another, for autotune and for the
//! candidates it measures.
//!
//! A key describes the *problem* — the head shape, the extents, which keys a
//! query sees and the scale its scores take — so it lives beside the plans
//! rather than beside the autotune that consumes it, the way
//! [`cubek_matmul::kernels::tune::key`] does. A caller builds the autotune; it is
//! this crate that knows what makes two attentions the same attention.
//!
//! One key for a prompt, a chunk of one and a decode step: the microkernel
//! family serves all three, so one key states the queries and the keys an autotune
//! is asked about, and each routine reads the problem it serves off it,
//! [`crate::kernels::AttentionRoutineProblem`]. It reads the key alone,
//! because the key is the whole of what cubecl hands a priority function.

use crate::kernels::{
    ATTENTION_GROUP_UNEVEN, AttentionCacheLayout, AttentionRoutineProblem, AttentionVisibility,
    QueryAxis, ScoreScale,
};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::attention::AttentionOperands;
use serde::{Deserialize, Serialize};

/// What makes two attentions want the same kernel.
///
/// The head shape is **exact** — it is fixed by the model config, and the GQA
/// ratio decides how much expansion a flash candidate pays. The queries, the
/// keys and the batch are anchored: they are the axes a prompt and a cache
/// vary freely, and where the crossovers between kernels live.
///
/// **A bucket's representative is the extent cubecl's anchor rounds a call up
/// to**, and a candidate's duplicate proof is taken there rather than at the
/// call in hand, so the candidates that collapse at some length inside a bucket
/// collapse together.
///
/// **No key reads a cache's size.** A cache's buffers hold its capacity, and how many of their
/// positions a call attends lives on device; neither is the key's to read. So two contexts, two
/// cache sizes or two devices sized apart tune the same attention under the same key, and an
/// environment built at one context serves every other. Where a call's buffers are its own keys,
/// every position of them attended ([`AttentionTuneInputs::attending_every_position`]), their
/// extent is the problem's, and the key holds it.
///
/// Four shapes reach autotune, and the key states each by its extents:
///
/// - **A whole prompt** over its own keys: its length is both the queries and the keys.
/// - **Queries over keys of their own** of another length, every position attended: both
///   extents, as the call brought them.
/// - **Queries over a cache**: a prompt, or a chunk of one prefilled in chunks, attended against
///   what the chunks before it filled. That is where a long prompt spends its attention, every
///   chunk after the first, so the key stands for
///   [`CHUNK_QUERIES`](Self::CHUNK_QUERIES) queries over [`CACHE_PREFIX`](Self::CACHE_PREFIX)
///   keys, at their buckets' extents, whatever the call brought.
/// - **A decode step over a cache**: one query over [`CACHE_PREFIX`](Self::CACHE_PREFIX) keys,
///   whatever the cache holds or the step attends. The attended count is a device tensor, so that
///   the launch is identical across steps, the graph-capture precondition.
///
/// [`AttentionTuneInputs::attending_every_position`]: crate::kernels::tune::AttentionTuneInputs::attending_every_position
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct AttentionKey {
    /// Query/key/value dtype.
    elem: ElemType,
    /// Query heads.
    heads: usize,
    /// Key/value heads (GQA when fewer than `heads`).
    kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Query positions.
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    queries: usize,
    /// Key positions: the buffers' extent where they are the call's own keys, the prefix every
    /// key over a cache stands for otherwise ([`CACHE_PREFIX`](Self::CACHE_PREFIX)), never a
    /// cache's capacity.
    #[autotune(anchor(exp(min = 1, max = 1048576, base = 2)))]
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
}

impl AttentionKey {
    /// The queries a key of many queries over a cache stands for: the narrowest chunk a prompt is
    /// cut into when it is prefilled in chunks. A device that cuts its prompts wider is keyed
    /// here all the same, so its chunks run what this chunk elected.
    pub const CHUNK_QUERIES: usize = 512;

    /// The attended prefix every key over a cache stands for, a chunk's and a decode step's alike:
    /// three chunks into its prompt, deep enough that the walk outweighs the launch. A cache of
    /// any capacity, attended to any length, is keyed here.
    pub const CACHE_PREFIX: usize = 4 * Self::CHUNK_QUERIES;

    /// Whether the queries are the tail of a longer attended prefix rather
    /// than the whole of it: a chunk, or a decode step.
    ///
    /// The flash routines ([`forward`](crate::forward)) serve only a whole prompt: they take no
    /// attended length, and their causal mask is top-left aligned, where a chunk's queries sit at
    /// the bottom right of the prefix. On a chunk they would be measuring a different function, so
    /// the flash candidates decline it. Both extents are anchored alike, so a whole prompt's are
    /// equal in every bucket.
    pub fn chunk(&self) -> bool {
        self.queries != self.keys
    }

    /// Query positions, at the bucket's own extent.
    pub fn seq_q(&self) -> usize {
        self.queries
    }

    /// Key positions, at the bucket's own extent.
    pub fn seq_kv(&self) -> usize {
        self.keys
    }

    /// The element the operands bind at.
    pub fn elem(&self) -> ElemType {
        self.elem
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
}

/// The problem a tuning **key** describes, at its bucket's extents
/// ([`AttentionKey`]): operands of the key's element, the query `[b, h, q, d]`, the keys and
/// values `[b, kv, s, d]`, read through a page table where the key has a page size, and the
/// output `[b, q, h, d]`, the value dim the head dim.
/// A key carries no value dim of its own: every attention reaching the table has `DV = D`.
impl TryFrom<&AttentionKey> for AttentionRoutineProblem {
    type Error = Refusal;

    /// # Errors
    ///
    /// [`ATTENTION_GROUP_UNEVEN`], where the key's query heads do not split evenly over its key
    /// heads.
    fn try_from(key: &AttentionKey) -> Result<Self, Refusal> {
        let group = key.group().ok_or(ATTENTION_GROUP_UNEVEN)?;
        let (element, head) = (key.elem(), key.head_dim());
        let rows = |innermost: usize| ElemAlignment::new(&[1, innermost], &[innermost, 1]);
        let (group, queries, visibility) = QueryAxis::new(key.seq_q(), key.visibility()).read(
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
                query: element,
                key: element,
                value: element,
                out: element,
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
        })
    }
}
