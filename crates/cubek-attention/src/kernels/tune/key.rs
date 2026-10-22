//! How an attention is told apart from another, for the tables and for the
//! rows they are built from.
//!
//! A key describes the *problem* — the head shape, the extents, which keys a
//! query sees and the scale its scores take — so it lives beside the plans
//! rather than beside the tables that consume it, the way
//! [`cubek_matmul::kernels::tune::key`] does. A caller builds the tables; it is
//! this crate that knows what makes two attentions the same attention.
//!
//! One key for a prompt, a chunk of one and a decode step: the microkernel
//! family serves all three, so one key states the queries and the keys a table
//! is asked about, and each routine reads the problem it serves off it,
//! [`AttentionRoutineProblem`](crate::kernels::AttentionRoutineProblem). It reads the key alone,
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
/// Three shapes reach the tables, and the key states each by its extents:
///
/// - **A whole prompt**: its length is both the queries and the keys.
/// - **A chunk** of a prompt prefilled in chunks, attended against the cache
///   the chunks before it filled: `queries` of the last of a longer prefix.
///   That is where a served prompt spends its attention, every chunk after
///   the first, and the walk an environment build takes lands every
///   `(queries, prefix)` bucket a context can produce, so a chunk's key stands
///   for [`CHUNK_QUERIES`](Self::CHUNK_QUERIES) queries over
///   [`CHUNK_PREFIX`](Self::CHUNK_PREFIX) keys, at their buckets' extents,
///   whatever the call brought.
/// - **A decode step**: one query against the cache's **capacity** — never the
///   attended count, which the launch takes as a device tensor so that it is
///   identical across steps, the graph-capture precondition.
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
    /// Key positions the buffers hold: a prompt's length, a chunk's prefix, a
    /// cache's capacity.
    #[autotune(anchor(exp(min = 1, max = 1048576, base = 2)))]
    keys: usize,
    #[autotune(anchor(exp(min = 1, max = 64, base = 2)))]
    batch: usize,
    /// Which keys a query sees: a causal mask and a window are part of the
    /// computation, not a detail of it, and cubek's flash routines serve
    /// neither a window nor a chunk's alignment.
    visibility: AttentionVisibility,
    /// Which scale the scores take: cubek's flash routines apply
    /// `1 / sqrt(head_dim)` and no other, so a winner measured at one cannot
    /// serve the other.
    scale: ScoreScale,
    /// Positions a page holds where the keys and values are read through a page table: a paged
    /// read costs what a dense one does not, so a winner measured on one does not serve the other.
    page_size: Option<usize>,
}

impl AttentionKey {
    /// The queries a chunk key stands for: the narrowest chunk a served prompt is cut into, the
    /// floor a server's prefill chunks are held to. A device that cuts its prompts wider is keyed
    /// here all the same, so its chunks run what this chunk elected.
    pub const CHUNK_QUERIES: usize = 512;

    /// The attended prefix a chunk key stands for: a chunk three chunks into
    /// its prompt, deep enough that the walk outweighs the launch.
    pub const CHUNK_PREFIX: usize = 4 * Self::CHUNK_QUERIES;

    /// Whether the queries are the tail of a longer attended prefix rather
    /// than the whole of it: a chunk, or a decode step.
    ///
    /// cubek's flash routines serve only a whole prompt: they take no attended
    /// length, and their causal mask is top-left aligned, where a chunk's
    /// queries sit at the bottom right of the prefix. On a chunk they would be
    /// measuring a different function, so the table's flash candidates decline
    /// it. Both extents are anchored alike, so a whole prompt's are equal in
    /// every bucket.
    pub fn chunk(&self) -> bool {
        self.queries != self.keys
    }

    /// Query positions, at the bucket's own extent.
    pub fn seq_q(&self) -> usize {
        self.queries
    }

    /// Key positions the buffers hold, at the bucket's own extent.
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
        let (group, queries, visibility) =
            QueryAxis::new(key.seq_q()).read(group, key.seq_q(), key.visibility());
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
