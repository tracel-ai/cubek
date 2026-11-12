//! The tensors of one attention, and how a launch binds them.

use crate::kernels::conversion::converted;
use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionCache, AttentionCacheBinding, AttentionQueries,
    AttentionRoutineProblem, AttentionScoring, KvSource, PAGED_PACKED_UNREAD, PackedAttentionCache,
    QueryAxis,
};
use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::TensorArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Delivery, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{
    Axis, Indexing, Launcher, TileSpec, TileTensorLaunch, launch::Grid as LaunchGrid,
};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, B, D, DV, G, KV, Q, S,
};
use cubek_std::layout::BindingOrder;
use cubek_std::routine::ADMITTED_PLAN_RUNS;
use cubek_std::routine::Grid;

/// What a launch declines before preparing anything: an element the routines do not take, or
/// operands of two elements other than an FP8 score's.
pub const ATTENTION_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel attention takes an f32, f16 or bf16 query, and keys and values each in \
          one of those or in e4m3",
};

/// What a packed launch declines before preparing anything: an operand packed at a width the
/// decode does not read, two, four and eight bits being the ones it does.
pub const PACKED_WIDTH_UNREAD: Refusal = Refusal::Declined {
    why: "the packed attention reads operands of two, four and eight bits",
};

/// What a launch declines before preparing anything: query heads that do not split evenly over
/// the key heads.
pub const ATTENTION_GROUP_UNEVEN: Refusal = Refusal::Declined {
    why: "the microkernel attention's query heads are a whole group per key head",
};

/// The tensors of one attention on one client: the query, the cache it reads, each sequence's
/// attended length, and the output.
///
/// Every operand is read where it lies, by its strides: the query's heads are viewed as a group
/// per key head, `[b, kv, g, q, d]`, and the output, allocated `[b, q, h, dv]` so that a token's
/// heads are adjacent, is written through the same view. At a single position the group's heads
/// lie along the query axis instead ([`QueryAxis`]). Nothing is copied. The problem the tensors
/// state is read off them ([`problem`](Self::problem)).
#[derive(Clone)]
pub struct AttentionTensors {
    client: Client,
    /// The query as the kernel reads it, `[b, kv, g, q, d]` over the caller's buffer.
    query: PlainTensor,
    kv: KvSource,
    /// The `u32` attended length of each sequence, `[b]`: on the device, so a captured launch
    /// replays as the cache grows.
    lengths: PlainTensor,
    /// Where every sequence's queries lie end to end: their `u32` cumulative counts `[b]`, and the
    /// most queries any sequence has, which the launch covers.
    query_runs: Option<(PlainTensor, usize)>,
    /// The output, `[b, q, h, dv]`.
    out: TensorHandle,
    /// The output as the kernel writes it, `[b, kv, g, q, dv]` over `out`'s buffer.
    out_view: PlainTensor,
    scoring: AttentionScoring,
    axis: QueryAxis,
    /// The `u32` counters the cubes of a box count themselves in at, held by the caller across
    /// launches ([`with_arrivals`](Self::with_arrivals)); `None` where each launch makes its own.
    arrivals: Option<PlainTensor>,
    /// Each query's logsumexp, `[b, h, q]`, and the same buffer as the kernel writes it,
    /// `[b, kv, g, q]`, where the caller asks for it ([`with_logsumexp`](Self::with_logsumexp)).
    logsumexp: Option<(TensorHandle, PlainTensor)>,
    /// Which keys each query of each head does not see, `[b, kv, g, q, s]` over the caller's
    /// buffer as the query is ([`with_mask`](Self::with_mask)); `None` where it sees every key
    /// its visibility admits.
    mask: Option<PlainTensor>,
}

impl core::fmt::Debug for AttentionTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the attention.
        f.debug_struct("AttentionTensors")
            .field("query", &self.query)
            .field("kv", &self.kv)
            .field("lengths", &self.lengths)
            .field("query_runs", &self.query_runs)
            .field("out", &self.out)
            .field("out_view", &self.out_view)
            .field("scoring", &self.scoring)
            .field("axis", &self.axis)
            .field("arrivals", &self.arrivals)
            .field("logsumexp", &self.logsumexp)
            .field("mask", &self.mask)
            .finish()
    }
}

impl AttentionTensors {
    /// Attention of `queries` over `kv`, each query scoring the keys as `scoring` says; the output
    /// `[b, q, h, dv]` allocated, `dv` the values' head dim. Every way of stating an attention
    /// comes through here: [`new`](Self::new), [`paged`](Self::paged) and
    /// [`packed`](Self::packed) are this call on one kind of source. A source listing blocks of
    /// keys is read as pages of one block each ([`KvSource::read_as_block_pages`]).
    ///
    /// Where the keys carry a QJL fix-up, the query is followed along its head dim by its
    /// projection, `d` twice the cache's head dim: the keys' lines run as far, their second half
    /// the fix-up ([`PackedStepReads`](cubek_microkernel::attention::plane::PackedStepReads)).
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`], [`PACKED_WIDTH_UNREAD`] and [`ATTENTION_GROUP_UNEVEN`]:
    /// what the routines decline whatever the device; [`PAGED_PACKED_UNREAD`] and what
    /// [`KvSource::read_as_block_pages`] declines: sources no routine reads; and
    /// [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does not run.
    ///
    /// # Panics
    ///
    /// Operands whose ranks, batch, head or key dims do not agree, a page table that is not `u32`
    /// of rank two with one row per sequence, pools that disagree on their pages, key heads or
    /// page size, an empty pool, page or table row, or a packed cache whose head dim is not the
    /// query's: no attention at all.
    pub fn attention(
        client: &Client,
        queries: AttentionQueries,
        kv: KvSource,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        let (query, lengths, query_runs) = match queries {
            AttentionQueries::Dense { query, lengths } => (query, lengths, None),
            AttentionQueries::Packed {
                query,
                ends,
                longest,
                lengths,
            } => {
                // Every sequence's queries end to end, `[Σ q_b, h, d]`, viewed as a query of each
                // sequence `ends` counts, `[b, h, Σ q_b, d]`, the batch at stride zero: each
                // sequence's run of it is placed by its entry of `ends` as the launch binds it.
                let (shape, strides) = (&query.binding.shape, &query.binding.strides);
                assert_eq!(shape.len(), 3, "packed queries are [Σ q_b, h, d]");
                assert!(
                    ends.elem == Self::COUNTER && ends.shape().rank() == 1,
                    "the ends of packed queries are u32 [b]"
                );
                let sequences = query.viewed(Metadata::new(
                    [ends.shape()[0], shape[1], shape[0], shape[2]],
                    [0, strides[1], strides[0], strides[2]],
                ));
                (sequences, lengths, Some((ends, longest)))
            }
        };
        let kv = kv.read_as_block_pages(client, scoring.visibility)?;
        match (&kv.cache, &kv.pages) {
            (AttentionCache::Float { key, value }, pages) => {
                Self::admit_float(client, &query, key, value, pages.as_ref())?
            }
            (AttentionCache::Packed(cache), None) => Self::admit_packed(client, &query, cache)?,
            (AttentionCache::Packed(_), Some(_)) => return Err(PAGED_PACKED_UNREAD),
        }
        let group = Self::group(&query, kv.num_kv_heads())?;
        Ok(Self::assemble(
            client, query, group, kv, lengths, scoring, query_runs,
        ))
    }

    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of `key`
    /// (`[b, kv, s, d]`) and `value` (`[b, kv, s, dv]`), the queries the last `q` positions of the
    /// attended keys, each scoring the keys as `scoring` says: [`attention`](Self::attention) over
    /// float keys and values laid out as sequences.
    ///
    /// # Errors
    ///
    /// As [`attention`](Self::attention).
    ///
    /// # Panics
    ///
    /// As [`attention`](Self::attention).
    pub fn new(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        let queries = AttentionQueries::Dense { query, lengths };
        let kv = KvSource::new(AttentionCache::Float { key, value });
        Self::attention(client, queries, kv, scoring)
    }

    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of the keys
    /// and values `pages` lays out in two pools of pages, `key` (`[pages, kv, page_size, d]`) and
    /// `value` (`[pages, kv, page_size, dv]`) ([`KvSource::pages`]): [`attention`](Self::attention)
    /// over float keys and values in pages. Each sequence holds as many keys as its row of the
    /// table has pages.
    ///
    /// # Errors
    ///
    /// As [`attention`](Self::attention).
    ///
    /// # Panics
    ///
    /// As [`attention`](Self::attention).
    pub fn paged(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        pages: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        let queries = AttentionQueries::Dense { query, lengths };
        let kv = KvSource::new(AttentionCache::Float { key, value }).paged(pages);
        Self::attention(client, queries, kv, scoring)
    }

    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of a packed
    /// cache, the queries the last `q` positions of the attended keys:
    /// [`attention`](Self::attention) over a packed cache laid out as sequences.
    ///
    /// # Errors
    ///
    /// As [`attention`](Self::attention).
    ///
    /// # Panics
    ///
    /// A cache [`PackedAttentionCache::new`] refuses, or as [`attention`](Self::attention).
    pub fn packed(
        client: &Client,
        query: PlainTensor,
        cache: PackedAttentionCache,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        let queries = AttentionQueries::Dense { query, lengths };
        let kv = KvSource::new(AttentionCache::Packed(cache));
        Self::attention(client, queries, kv, scoring)
    }

    /// The elements the routines are verified at.
    pub const VERIFIED: [ElemType; 3] = [
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
    ];

    /// The elements a side of a cache is verified at besides [`VERIFIED`](Self::VERIFIED): an
    /// FP8 cache's, which the routines read on the device's FP8 instruction or widen as they
    /// stage it.
    pub const VERIFIED_CACHE: [ElemType; 1] = [ElemType::Float(FloatKind::E4M3)];

    /// Float operands of one element the routines take and `client`'s device runs, `key` and
    /// `value` laid out as sequences, or as pools of pages where `pages` names them.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`] and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// As [`attention`](Self::attention), for float keys and values.
    fn admit_float(
        client: &Client,
        query: &PlainTensor,
        key: &PlainTensor,
        value: &PlainTensor,
        pages: Option<&PlainTensor>,
    ) -> Result<(), Refusal> {
        Self::admit_elements(client, query, key, value)?;
        let (q_shape, k_shape, v_shape) = (query.shape(), key.shape(), value.shape());
        assert!(
            q_shape.num_dims() == 4 && k_shape.num_dims() == 4 && v_shape.num_dims() == 4,
            "attention's operands are [b, h, q, d], and [b, kv, s, ·] or [pages, kv, page_size, ·]"
        );
        assert!(
            v_shape[0] == k_shape[0] && v_shape[1] == k_shape[1] && v_shape[2] == k_shape[2],
            "the keys and values agree on their sequences or pages, key heads and positions"
        );
        assert_eq!(k_shape[3], q_shape[3], "the keys' head dim is the query's");
        match pages {
            Some(pages) => {
                Self::assert_page_table(pages, q_shape[0]);
                assert!(
                    k_shape[0] > 0 && k_shape[2] > 0 && pages.shape()[1] > 0,
                    "a paged attention reads a pool, its pages and its table rows non-empty"
                );
            }
            None => assert_eq!(k_shape[0], q_shape[0], "the operands' batch dims agree"),
        }
        Ok(())
    }

    /// A packed cache the routines read, under `query` of an element they take and `client`'s
    /// device runs.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`], [`ELEMENT_NOT_ON_DEVICE`] and [`PACKED_WIDTH_UNREAD`].
    ///
    /// # Panics
    ///
    /// A query whose head dim is not the cache's, twice it with a fix-up.
    fn admit_packed(
        client: &Client,
        query: &PlainTensor,
        cache: &PackedAttentionCache,
    ) -> Result<(), Refusal> {
        Self::admit_element(client, query)?;
        if ![cache.bits.key, cache.bits.value]
            .iter()
            .all(|bits| matches!(bits, 2 | 4 | 8))
        {
            return Err(PACKED_WIDTH_UNREAD);
        }
        let coords = cache.coords();
        let widened = match cache.fixup {
            Some(_) => 2 * coords,
            None => coords,
        };
        assert_eq!(query.shape()[3], widened, "the query spans the keys' lines");
        Ok(())
    }

    /// The elements of float operands, where the routines take them and `client`'s device runs
    /// them: the query in an element the device computes in, and the keys and the values each in
    /// one of those or in `e4m3` ([`VERIFIED_CACHE`](Self::VERIFIED_CACHE)), whatever the other
    /// side's, where the device stores it. A routine reads each side as stored or converted as
    /// it stages it ([`AttentionWorkItem::readings`](cubek_microkernel::attention::AttentionWorkItem::readings)),
    /// and declines a pair it reads neither way.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`] and [`ELEMENT_NOT_ON_DEVICE`].
    fn admit_elements(
        client: &Client,
        query: &PlainTensor,
        key: &PlainTensor,
        value: &PlainTensor,
    ) -> Result<(), Refusal> {
        Self::admit_element(client, query)?;
        for side in [key, value] {
            let verified =
                Self::VERIFIED.contains(&side.elem) || Self::VERIFIED_CACHE.contains(&side.elem);
            if !verified {
                return Err(ATTENTION_ELEMENT_UNVERIFIED);
            }
            let stored = client
                .properties()
                .features
                .type_usage(side.elem)
                .contains(TypeUsage::Buffer);
            if !stored {
                return Err(ELEMENT_NOT_ON_DEVICE);
            }
        }
        Ok(())
    }

    /// The query's element, where the routines take it and `client`'s device runs it.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`] and [`ELEMENT_NOT_ON_DEVICE`].
    fn admit_element(client: &Client, query: &PlainTensor) -> Result<(), Refusal> {
        if !Self::VERIFIED.contains(&query.elem) {
            return Err(ATTENTION_ELEMENT_UNVERIFIED);
        }
        if !query.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        Ok(())
    }

    /// Query heads per key head, of `query`'s heads over `kv_heads`.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_GROUP_UNEVEN`].
    fn group(query: &PlainTensor, kv_heads: usize) -> Result<usize, Refusal> {
        let heads = query.shape()[1];
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(ATTENTION_GROUP_UNEVEN);
        }
        Ok(heads / kv_heads)
    }

    /// The tensors of an attention of `query`, `group` heads a key head, over `kv`: the output
    /// allocated, and the query and the output viewed along the query axis. Where `query_runs`
    /// says every sequence's queries lie end to end, the output does too, `[Σ q_b, h, dv]`, viewed
    /// as the query is, and the queries are read along
    /// their positions.
    fn assemble(
        client: &Client,
        query: PlainTensor,
        group: usize,
        kv: KvSource,
        lengths: PlainTensor,
        scoring: AttentionScoring,
        query_runs: Option<(PlainTensor, usize)>,
    ) -> Self {
        let q_shape = query.shape();
        let (batch, heads, queries) = (q_shape[0], q_shape[1], q_shape[2]);
        let value = kv.val_dim();
        // The output is at the query's element, the one the model computes in, whatever each
        // side of the cache is stored in.
        let out_elem = query.elem;
        // Dense, as every arm of the attention seam allocates it: a token's heads are one row, so
        // the caller merges them by a reshape. The views below are a factorization of these
        // strides, which a pitched allocation would invalidate.
        let (axis, out, out_plain) = match &query_runs {
            None => {
                let (out, out_plain) = PlainTensor::empty_contiguous(
                    client,
                    [batch, queries, heads, value].into(),
                    out_elem,
                );
                (QueryAxis::new(queries, scoring.visibility), out, out_plain)
            }
            Some(_) => {
                let (out, end_to_end) =
                    PlainTensor::empty_contiguous(client, [queries, heads, value].into(), out_elem);
                let strides = &end_to_end.binding.strides;
                let out_plain = end_to_end.viewed(Metadata::new(
                    [batch, queries, heads, value],
                    [0, strides[0], strides[1], strides[2]],
                ));
                (QueryAxis::Positions, out, out_plain)
            }
        };
        Self {
            client: client.clone(),
            query: axis.view(&query, group, [0, 1, 2, 3]),
            out_view: axis.view(&out_plain, group, [0, 2, 1, 3]),
            out,
            kv,
            lengths: Self::one_per_sequence(lengths, batch),
            query_runs,
            scoring,
            axis,
            arrivals: None,
            logsumexp: None,
            mask: None,
        }
    }

    /// `lengths` as one `u32` value per sequence, `(sequences,)`, read by stride: a `(1,)`
    /// tensor serves every sequence as a stride-0 view of its one element rather than a copy, so
    /// the kernel still reads the buffer an advance moves in place.
    ///
    /// # Panics
    ///
    /// If it is not `u32`, or holds neither one value nor one per sequence.
    fn one_per_sequence(lengths: PlainTensor, sequences: usize) -> PlainTensor {
        assert_eq!(
            lengths.elem,
            Self::COUNTER,
            "a per-sequence length is u32, got {:?}",
            lengths.elem
        );
        match lengths.shape().as_slice() {
            [1] => lengths.viewed(Metadata::new([sequences], [0])),
            [held] if *held == sequences => lengths,
            [held] => panic!("expected one value, or one per sequence ({sequences}), got {held}"),
            dims => panic!("a per-sequence length is of rank one, got {dims:?}"),
        }
    }

    /// `pages` is a page table of `sequences`: `u32` `(sequences, pages_per_sequence)`, read by
    /// stride, so it is not copied.
    ///
    /// # Panics
    ///
    /// If it is not `u32`, not rank 2, or holds another count of rows.
    fn assert_page_table(pages: &PlainTensor, sequences: usize) {
        assert_eq!(
            pages.elem,
            Self::COUNTER,
            "a page table is u32, got {:?}",
            pages.elem
        );
        let shape = pages.shape();
        assert_eq!(shape.rank(), 2, "a page table is of rank two");
        assert_eq!(
            shape[0], sequences,
            "a page table holds one row per sequence"
        );
    }

    /// The element of every count the kernels read: a length, a page, an arrival.
    const COUNTER: ElemType = ElemType::UInt(UIntKind::U32);

    /// The same attention, its cubes counting themselves in at `arrivals`, `u32` counters at zero
    /// that the caller keeps from one launch to the next: every launch leaves them at zero, so a
    /// launch that keeps them spends no fill on them. As many as the boxes of the largest problem
    /// the caller attends, or more; `None` leaves each launch to fill its own.
    ///
    /// # Panics
    ///
    /// Counters that are not `u32`.
    pub fn with_arrivals(mut self, arrivals: Option<PlainTensor>) -> Self {
        if let Some(kept) = &arrivals {
            assert_eq!(kept.elem, Self::COUNTER, "arrival counters are u32");
        }
        self.arrivals = arrivals;
        self
    }

    /// The same attention, query `q` of head `h` of sequence `b` not seeing key `s` where
    /// `mask[b, h, q, s]` is not zero, on top of what its visibility hides. A dim the mask does
    /// not vary along is read at stride zero, so a mask broadcast over the batch or the heads is
    /// never copied. Its element is any integer, a stored boolean among them.
    ///
    /// # Panics
    ///
    /// A mask of another shape than the scores', `[b, h, q, s]`.
    pub fn with_mask(mut self, mask: PlainTensor) -> Self {
        assert!(
            self.query_runs.is_none(),
            "queries that lie end to end take no mask"
        );
        let [batch, queries, heads, _] = self.out.shape().dims::<4>();
        assert_eq!(
            mask.shape().as_slice(),
            [batch, heads, queries, self.kv.seq_kv()],
            "a mask is [b, h, q, s], one cell a score"
        );
        let group = heads / self.kv.num_kv_heads();
        self.mask = Some(self.axis.view(&mask, group, [0, 1, 2, 3]));
        self
    }

    /// The same attention, writing beside its output each query's logsumexp of its scaled scores,
    /// `[b, h, q]` in the sum's element ([`logsumexp`](Self::logsumexp)): what a backward pass reads
    /// its probabilities back from. A query that sees no key holds the minimum value.
    ///
    /// # Panics
    ///
    /// On a packed cache, which is read for inference: no backward pass reads its logsumexp.
    pub fn with_logsumexp(mut self) -> Self {
        assert!(
            !matches!(self.kv.cache, AttentionCache::Packed(_)),
            "a packed cache's attention writes no logsumexp"
        );
        assert!(
            self.query_runs.is_none(),
            "queries that lie end to end write no logsumexp"
        );
        let [batch, queries, heads, _] = self.out.shape().dims::<4>();
        let (handle, rows) = PlainTensor::empty_contiguous(
            &self.client,
            [batch, heads, queries].into(),
            AttentionRoutineProblem::SUM,
        );
        // Viewed along the query axis as the output is: a last dim one wide, then left out.
        let strides = &rows.binding.strides;
        let widened = rows.viewed(Metadata::new(
            [batch, heads, queries, 1],
            [strides[0], strides[1], strides[2], 1],
        ));
        let group = heads / self.kv.num_kv_heads();
        let view = self.axis.view(&widened, group, [0, 1, 2, 3]);
        let (shape, strides) = (&view.binding.shape, &view.binding.strides);
        let view = view.viewed(Metadata::new(shape[..4].to_vec(), strides[..4].to_vec()));
        self.logsumexp = Some((handle, view));
        self
    }

    /// Each query's logsumexp, `[b, h, q]`, where the attention writes it
    /// ([`with_logsumexp`](Self::with_logsumexp)).
    pub fn logsumexp(&self) -> Option<TensorHandle> {
        self.logsumexp.as_ref().map(|(handle, _)| handle.clone())
    }

    /// Each query's logsumexp, `[b, kv, g, q]` at its own strides, bound as a plain tensor where
    /// the attention writes it: what a kernel that writes it by hand addresses.
    pub fn logsumexp_arg(&self) -> Option<TensorArg> {
        self.logsumexp
            .as_ref()
            .map(|(_, view)| view.binding.clone().into_tensor_arg())
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> AttentionRoutineProblem {
        let [batch, queries, heads, value] = match &self.query_runs {
            None => self.out.shape().dims::<4>(),
            Some((ends, longest)) => {
                let [_, heads, value] = self.out.shape().dims::<3>();
                [ends.shape()[0], *longest, heads, value]
            }
        };
        let kv_heads = self.kv.num_kv_heads();
        let (group, queries, visibility) =
            self.axis
                .read(heads / kv_heads, queries, self.scoring.visibility);
        let head = self.query.shape()[4];
        let start = PlainTensor::start_bytes;
        let query_elem = self.query.elem;
        let (key_alignment, value_alignment, key_start, value_start, key_elem, value_elem) =
            match &self.kv.cache {
                AttentionCache::Float { key, value } => (
                    key.elem_alignment(),
                    value.elem_alignment(),
                    start(key),
                    start(value),
                    key.elem,
                    value.elem,
                ),
                // The decoded lines lie as dense rows of the head dim, at the query's element.
                AttentionCache::Packed(_) => (
                    ElemAlignment::dense_rows(head),
                    ElemAlignment::dense_rows(value),
                    0,
                    0,
                    query_elem,
                    query_elem,
                ),
            };
        AttentionRoutineProblem {
            batch,
            num_kv_heads: kv_heads,
            group,
            seq_q: queries,
            seq_kv: self.kv.seq_kv(),
            head_dim: head,
            val_dim: value,
            visibility,
            elements: AttentionOperands {
                query: query_elem,
                key: key_elem,
                value: value_elem,
                out: self.out_view.elem,
            },
            elem_alignments: AttentionOperands {
                query: self.query.elem_alignment(),
                key: key_alignment,
                value: value_alignment,
                out: self.out_view.elem_alignment(),
            },
            start_bytes: AttentionOperands {
                query: start(&self.query),
                key: key_start,
                value: value_start,
                out: start(&self.out_view),
            },
            cache: self.kv.layout(),
            packed_queries: self.query_runs.is_some(),
        }
    }

    /// Where the keys and values are read from.
    pub fn kv(&self) -> &KvSource {
        &self.kv
    }

    /// The client the operands live on, which a launch submits to.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The scale each dot product takes before the softmax: a launch argument, no part of the
    /// problem a routine plans.
    pub fn scale(&self) -> f32 {
        self.scoring.scale.value()
    }

    /// The output, `[b, q, h, dv]`; `[Σ q_b, h, dv]` where the queries lie end to end.
    pub fn into_out(self) -> TensorHandle {
        self.out
    }

    /// The output, `[b, kv, g, q, dv]` at its own strides, bound as a plain tensor: what a kernel
    /// that writes it by hand addresses.
    pub fn out_arg(&self) -> TensorArg {
        self.out_view.binding.clone().into_tensor_arg()
    }

    /// The operands bound as `blueprint` states them, on `grid`: under its partitioning, at its
    /// widths, the keys in the order its microkernel reads them, and the query, the keys and the
    /// values read at the element its microkernel serves them in
    /// ([`AttentionMicrokernel::served`]). A packed cache binds no tile: its kernel decodes it as
    /// it reads it ([`AttentionCacheBinding::Packed`]).
    pub fn bind<M: AttentionMicrokernel, SM>(
        &self,
        blueprint: &AttentionBlueprint<M, SM>,
        grid: &Grid,
    ) -> AttentionBinding {
        let partitioning = &blueprint.partitioning;
        let launcher = Launcher::new(
            &self.client,
            partitioning.kernel(),
            partitioning.as_tile().space(),
            LaunchGrid::Stated {
                cube_count: grid.launch.cube_count.clone(),
                cube_dim: grid.launch.cube_dim,
            },
        )
        .expect(ADMITTED_PLAN_RUNS);
        let widths = blueprint.widths;
        let problem = self.problem();
        let read = blueprint.microkernel.served(&problem.work_item()).elements;
        // Read at another element than the float it is stored in, an operand each of whose values
        // is a word of its own is bound as its u32 words, which its kernel's stages decode at the
        // read element. A query of narrower values is converted ahead, into a buffer of its own
        // (`converted`); the microkernels read no narrower side of a cache converted.
        let words = |operand: AttentionOperand| match problem.elements.get(operand) {
            stored @ ElemType::Float(kind)
                if read.get(operand) != stored && stored.size() == u32::BITS as usize / 8 =>
            {
                Some(kind)
            }
            _ => None,
        };
        let converted_query =
            read.query != problem.elements.query && words(AttentionOperand::Query).is_none();
        let query = match converted_query {
            true => converted(&self.client, &self.query, read.query),
            false => self.query.clone(),
        };
        let arg = |tensor: &PlainTensor, axes: &'static [Axis], width: usize, element: ElemType| {
            launcher
                .arg(tensor.binding.clone(), element)
                .axes(axes)
                .vectorize(width)
        };
        let read_operand =
            |tensor: &PlainTensor,
             axes: &'static [Axis],
             width: usize,
             operand: AttentionOperand| match words(operand) {
                Some(kind) => arg(tensor, axes, width, ElemType::UInt(UIntKind::U32)).packed(kind),
                None => arg(tensor, axes, width, read.get(operand)),
            };
        let query_delivery = match words(AttentionOperand::Query) {
            Some(_) => Delivery::SyncPerUnit,
            None => blueprint
                .microkernel
                .query_delivery(&widths, &problem.work_item()),
        };
        // Where the queries lie end to end, the query and the output each read the run of their
        // sequence: its entry of the cumulative counts, keyed by the batch, places it along `Q`.
        let runs = |bound: TileTensorLaunch| match &self.query_runs {
            Some((ends, _)) => bound
                .indexed(
                    TileTensorLaunch::new(
                        ends.binding.clone().into_tensor_arg(),
                        Self::COUNTER,
                        1,
                        TileSpec::direct(&[B]),
                    ),
                    Indexing::Run(Q),
                )
                .expect(ADMITTED_PLAN_RUNS),
            None => bound,
        };
        let query = runs(
            read_operand(
                &query,
                &[B, KV, G, Q, D],
                widths.query,
                AttentionOperand::Query,
            )
            .delivery(query_delivery)
            .build()
            .expect(ADMITTED_PLAN_RUNS)
            .arg(),
        );
        // The keys and values move into their stages as the microkernel says, unless they are
        // bound as words its stages decode, which pass through the units' registers.
        let delivery = match words(AttentionOperand::Key).or(words(AttentionOperand::Value)) {
            Some(_) => Delivery::SyncPerUnit,
            None => blueprint
                .microkernel
                .delivery(&widths, &problem.work_item()),
        };
        let cache = match &self.kv.cache {
            AttentionCache::Float { key, value } => {
                // A pool of pages is placed by its table, keyed by the batch and the keys'
                // positions: it covers the launch, its rows the batch and its columns the keys.
                let read_cache = |tensor: &PlainTensor,
                                  axes: &'static [Axis],
                                  width: usize,
                                  operand: AttentionOperand| {
                    let arg = read_operand(tensor, axes, width, operand)
                        .delivery(delivery)
                        .build()
                        .expect(ADMITTED_PLAN_RUNS)
                        .arg();
                    match &self.kv.pages {
                        Some(pages) => arg
                            .indexed_covering(
                                TileTensorLaunch::new(
                                    pages.binding.clone().into_tensor_arg(),
                                    ElemType::UInt(UIntKind::U32),
                                    1,
                                    TileSpec::direct(&[B, S]),
                                ),
                                Indexing::Coordinate(B),
                            )
                            .expect(ADMITTED_PLAN_RUNS),
                        None => arg,
                    }
                };
                let key = match blueprint.microkernel.key_order() {
                    BindingOrder::InBufferOrder => {
                        read_cache(key, &[B, KV, S, D], widths.key, AttentionOperand::Key)
                    }
                    BindingOrder::Stated => read_cache(
                        &key.transposed(),
                        &[B, KV, D, S],
                        widths.key,
                        AttentionOperand::Key,
                    ),
                };
                let value = read_cache(
                    value,
                    &[B, KV, S, DV],
                    widths.value,
                    AttentionOperand::Value,
                );
                AttentionCacheBinding::Tiles { key, value }
            }
            AttentionCache::Packed(cache) => AttentionCacheBinding::Packed(cache.bind(&launcher)),
        };
        let out = runs(
            arg(
                &self.out_view,
                &[B, KV, G, Q, DV],
                widths.out,
                problem.elements.out,
            )
            .build()
            .expect(ADMITTED_PLAN_RUNS)
            .arg(),
        );
        // A value a query, written by whoever holds its row.
        let logsumexp = self.logsumexp.as_ref().map(|(_, view)| {
            arg(view, &[B, KV, G, Q], 1, AttentionRoutineProblem::SUM)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        });
        // A cell a read: a mask broadcast over a dim lies there at stride zero.
        let mask = self.mask.as_ref().map(|mask| {
            arg(mask, &[B, KV, G, Q, S], 1, mask.elem)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        });
        AttentionBinding {
            launcher,
            query,
            cache,
            out,
            logsumexp,
            mask,
            lengths: self.lengths.binding.clone().into_tensor_arg(),
            query_ends: self
                .query_runs
                .as_ref()
                .map(|(ends, _)| ends.binding.clone().into_tensor_arg()),
        }
    }

    /// The slots the cubes sharing a box of queries leave their partials in, one for each of the
    /// `splits` cubes.
    pub fn split_slots(&self, splits: usize) -> AttentionSplitSlots {
        let problem = self.problem();
        let slots = [
            problem.batch,
            problem.num_kv_heads,
            problem.group,
            problem.seq_q,
            splits,
        ];
        let output = |shape: &[usize]| {
            let shape = shape.into();
            PlainTensor::empty_contiguous(&self.client, shape, AttentionRoutineProblem::SUM).0
        };
        AttentionSplitSlots {
            out: output(&[slots.as_slice(), &[problem.val_dim]].concat()),
            max: output(&slots),
            sum: output(&slots),
        }
    }

    /// A counter for every box of `box_queries` queries, holding zero: where the cubes sharing a
    /// box count themselves in. The caller's ([`with_arrivals`](Self::with_arrivals)) where it
    /// keeps them, otherwise filled for this launch.
    ///
    /// # Panics
    ///
    /// Kept counters fewer than the boxes.
    pub fn box_arrivals(&self, box_queries: usize) -> PlainTensor {
        let problem = self.problem();
        let boxes = problem.batch
            * problem.num_kv_heads
            * problem.group
            * problem.seq_q.div_ceil(box_queries);
        match &self.arrivals {
            Some(kept) => {
                let counters = kept.shape().num_elements();
                assert!(
                    counters >= boxes,
                    "the kept arrival counters cover every box: {boxes} boxes, {counters} counters"
                );
                kept.clone()
            }
            None => {
                PlainTensor::of_handle(TensorHandle::zeros(&self.client, [boxes], Self::COUNTER))
            }
        }
    }
}

/// Where the cubes sharing a box of queries leave their partials, in the sum's element
/// ([`AttentionTensors::split_slots`]).
pub struct AttentionSplitSlots {
    /// Each query's unnormalized output in every slot, `[b, kv, g, q, splits, dv]`.
    pub out: TensorHandle,
    /// Each query's running max in every slot, `[b, kv, g, q, splits]`.
    pub max: TensorHandle,
    /// Each query's running sum of probabilities in every slot, beside the max.
    pub sum: TensorHandle,
}
