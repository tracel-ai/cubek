//! The tensors of one attention, and how a launch binds them.

#[cfg(doc)]
use crate::kernels::CACHE_SEQUENCES_DISAGREE;
use crate::kernels::conversion::{converted, distinct_elements};
use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionCache, AttentionCacheBinding, AttentionQueries,
    AttentionRoutineProblem, AttentionScoring, KvSource, PackedAttentionCache, QueryDimension,
    SCALED_PACKED_UNREAD, page_table,
};
use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::TensorArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Bound, Decline, Delivery};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::{
    Dimension, Indexing, Launcher, Projection, TileSpec, TileTensorLaunch,
    launch::Grid as LaunchGrid,
};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, B, CacheLines, D, DV, G, KV, Q, S,
};
use cubek_std::routine::Grid;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable};

/// What a launch declines before preparing anything: an element the routines do not take, or
/// operands of two elements other than an FP8 score's.
pub const ATTENTION_ELEMENT_UNVERIFIED: Decline = Decline::Unserved {
    why: "the microkernel attention takes an f32, f16 or bf16 query, and keys and values each in \
          one of those or in e4m3",
};

/// What a packed launch declines before preparing anything: an operand packed at a width the
/// decode does not read, two, four and eight bits being the ones it does.
pub const PACKED_WIDTH_UNREAD: Decline = Decline::Unserved {
    why: "the packed attention reads operands of two, four and eight bits",
};

/// What a launch declines before preparing anything: query heads that do not split evenly over
/// the key heads.
pub const ATTENTION_GROUP_UNEVEN: Decline = Decline::Unserved {
    why: "the microkernel attention's query heads are a whole group per key head",
};

/// What a launch declines before preparing anything: queries that are no `[b, h, q, d]`, or, lying
/// end to end, no `[Σ q_b, h, d]` under `u32` ends `[b]`, or whose `d` is not what the cache's keys
/// are scored over.
pub const ATTENTION_QUERIES_UNREAD: Decline = Decline::Unserved {
    why: "queries are [b, h, q, d], or [Σ q_b, h, d] under u32 ends [b], d the keys' head dim, \
          twice it under a QJL fix-up",
};

/// What a launch declines before preparing anything: more query elements than a `u32` counts.
pub const ATTENTION_QUERIES_UNCOUNTED: Decline = Decline::Unserved {
    why: "a query's elements are counted in a u32",
};

/// What a launch declines before preparing anything: attended lengths other than one `u32`, or one
/// a sequence.
pub const ATTENTION_LENGTHS_UNREAD: Decline = Decline::Unserved {
    why: "attended lengths are u32 of rank one, one value or one a sequence",
};

/// What an attention declines of a mask: another shape than the scores', or queries that lie end
/// to end.
pub const ATTENTION_MASK_UNREAD: Decline = Decline::Unserved {
    why: "a mask is [b, h, q, s], one cell a score, over queries [b, h, q, d]",
};

/// What an attention declines of a logsumexp: one over a packed cache, or of queries that lie end
/// to end.
pub const ATTENTION_LOGSUMEXP_UNWRITTEN: Decline = Decline::Unserved {
    why: "an attention writes its logsumexp over a float cache, for queries [b, h, q, d]",
};

/// The tensors of one attention on one client: the query, the cache it reads, each sequence's
/// attended length, and the output.
///
/// Every operand is read where it lies, by its strides: the query's heads are viewed as a group per
/// key head, `[b, kv, g, q, d]`, and the output, allocated `[b, q, h, dv]` so that a token's heads
/// are adjacent, is written through the same view. At a single position the group's heads lie along
/// the query dimension instead ([`QueryDimension`]). Nothing is copied. The problem the tensors
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
    dimension: QueryDimension,
    /// The `u32` counters the cubes of a box count themselves in at, kept by the caller across
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
            .field("dimension", &self.dimension)
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
    /// # Assumptions
    ///
    /// No run of queries lying end to end is longer than `longest`.
    ///
    /// # Guarantees
    ///
    /// Where every entry of a page or block table names a page or block there is, and the ends of
    /// queries lying end to end never decrease and end at their count, a launch writes the
    /// attention the operands state, an attended length past the keys a sequence holds attending as
    /// a length of the keys there are. Whatever the lengths, the entries and the ends hold, no
    /// launch reads or writes past an operand's buffer: an entry past the pool reads its last page,
    /// a listed block past its sequence's pages lies in their last page, an end past the queries'
    /// count is read as the count, and a start past its end as an empty run. A decline comes before
    /// anything is allocated or launched.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`], [`PACKED_WIDTH_UNREAD`] and [`ATTENTION_GROUP_UNEVEN`]:
    /// what the routines decline whatever the device; [`SCALED_PACKED_UNREAD`] and what
    /// [`KvSource::read_as_block_pages`] declines: sources no routine reads;
    /// [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does not run; and
    /// [`ATTENTION_QUERIES_UNREAD`], [`ATTENTION_QUERIES_UNCOUNTED`],
    /// [`ATTENTION_LENGTHS_UNREAD`] and [`CACHE_SEQUENCES_DISAGREE`]: operands that state no
    /// attention.
    pub fn attention(
        client: &Client,
        queries: AttentionQueries,
        kv: KvSource,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
        let (query, lengths, query_runs) = match queries {
            AttentionQueries::Dense { query, lengths } => (query, lengths, None),
            AttentionQueries::Packed {
                query,
                ends,
                longest,
                lengths,
            } => (
                Self::end_to_end(query, &ends)?,
                lengths,
                Some((ends, longest)),
            ),
        };
        if query.shape().rank() != 4 {
            return Err(ATTENTION_QUERIES_UNREAD);
        }
        if u32::try_from(distinct_elements(&query)).is_err() {
            return Err(ATTENTION_QUERIES_UNCOUNTED);
        }
        let pools = kv.block_pools(scoring.visibility)?;
        match &kv.cache {
            AttentionCache::Float { key, value } => Self::admit_float(client, &query, key, value)?,
            AttentionCache::Packed(_) if kv.scales.any() => return Err(SCALED_PACKED_UNREAD),
            AttentionCache::Packed(cache) => Self::admit_packed(client, &query, cache)?,
        }
        let sequences = query.shape()[0];
        kv.admit_sequences(sequences)?;
        let lengths = Self::one_per_sequence(lengths, sequences)?;
        let group = Self::group(&query, kv.num_kv_heads())?;
        let kv = kv.into_block_pages(client, pools);
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
    pub fn new(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
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
    pub fn paged(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        pages: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
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
    pub fn packed(
        client: &Client,
        query: PlainTensor,
        cache: PackedAttentionCache,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
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

    /// The element of every count the kernels read: a length, a page, an arrival.
    const COUNTER: ElemType = ElemType::UInt(UIntKind::U32);

    /// The same attention, its cubes counting themselves in at `arrivals`, `u32` counters at zero
    /// that the caller keeps from one launch to the next: every launch leaves them at zero, so a
    /// launch that keeps them spends no fill on them. As many as the boxes of the largest problem
    /// the caller attends, or more.
    ///
    /// # Assumptions
    ///
    /// The counters are zero when a launch starts, and no other launch counts at them meanwhile.
    ///
    /// # Guarantees
    ///
    /// A launch fills counters of its own where it is given `None`, or counters that are not `u32`
    /// or are fewer than its boxes.
    pub fn with_arrivals(mut self, arrivals: Option<PlainTensor>) -> Self {
        self.arrivals = arrivals;
        self
    }

    /// The same attention, query `q` of head `h` of sequence `b` not seeing key `s` where
    /// `mask[b, h, q, s]` is not zero, on top of what its visibility hides. A dim the mask does
    /// not vary along is read at stride zero, so a mask broadcast over the batch or the heads is
    /// never copied.
    ///
    /// # Assumptions
    ///
    /// The mask's element is an integer, a stored boolean among them.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_MASK_UNREAD`].
    pub fn with_mask(mut self, mask: PlainTensor) -> Result<Self, Decline> {
        if self.query_runs.is_some() {
            return Err(ATTENTION_MASK_UNREAD);
        }
        let [batch, queries, heads, _] = self.out.shape().dims::<4>();
        if mask.shape().as_slice() != [batch, heads, queries, self.kv.seq_kv()] {
            return Err(ATTENTION_MASK_UNREAD);
        }
        let group = heads / self.kv.num_kv_heads();
        self.mask = Some(self.dimension.view(&mask, group, [0, 1, 2, 3]));
        Ok(self)
    }

    /// The same attention, writing beside its output each query's logsumexp of its scaled scores,
    /// `[b, h, q]` in the sum's element ([`logsumexp`](Self::logsumexp)): what a backward pass
    /// reads its probabilities back from. A query that sees no key holds the minimum value.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_LOGSUMEXP_UNWRITTEN`]. A packed cache is read for inference: no backward pass
    /// reads its logsumexp.
    pub fn with_logsumexp(mut self) -> Result<Self, Decline> {
        if self.query_runs.is_some() || matches!(self.kv.cache, AttentionCache::Packed(_)) {
            return Err(ATTENTION_LOGSUMEXP_UNWRITTEN);
        }
        let [batch, queries, heads, _] = self.out.shape().dims::<4>();
        let (handle, rows) = PlainTensor::empty_contiguous(
            &self.client,
            [batch, heads, queries].into(),
            AttentionRoutineProblem::SUM,
        );
        // Viewed along the query dimension as the output is: a last dim one wide, then left out.
        let strides = &rows.binding.strides;
        let widened = rows.viewed(Metadata::new(
            [batch, heads, queries, 1],
            [strides[0], strides[1], strides[2], 1],
        ));
        let group = heads / self.kv.num_kv_heads();
        let view = self.dimension.view(&widened, group, [0, 1, 2, 3]);
        let (shape, strides) = (&view.binding.shape, &view.binding.strides);
        let view = view.viewed(Metadata::new(shape[..4].to_vec(), strides[..4].to_vec()));
        self.logsumexp = Some((handle, view));
        Ok(self)
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
            self.dimension
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
    /// ([`AttentionMicrokernel::served`]), each side of a float cache under its per-tensor scale
    /// ([`KvSource::scales`]). A packed cache binds no tile: its kernel decodes it as it reads it
    /// ([`AttentionCacheBinding::Packed`]).
    ///
    /// # Assumptions
    ///
    /// `blueprint` and `grid` were prepared for [`problem`](Self::problem).
    ///
    /// # Guarantees
    ///
    /// A Broken binding launches nothing but a conversion of the query, read at another element,
    /// where the query's own binding is the one refused.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    pub fn bind<M: AttentionMicrokernel, SM>(
        &self,
        blueprint: &AttentionBlueprint<M, SM>,
        grid: &Grid,
    ) -> Result<AttentionBinding, Decline> {
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
        .or_broken(ADMITTED_PLAN_RUNS)?;
        let widths = blueprint.widths;
        let problem = self.problem();
        let read = blueprint.microkernel.served(&problem.work_item()).elements;
        // Read at another element than the float it is stored in, a side of the cache is bound as
        // its u32 words, which its stage decodes at the read element as it fills: the cache is
        // read once a launch, where it lies. A decoding fill passes the words through the units'
        // registers, so such an operand moves into its stage per unit, whatever the microkernel
        // plans for the others; the stages meet on a barrier where two operands' moves differ.
        // The query is bound so only where each of its values is a word of its own. A query of
        // narrower values is a few megabytes against a cache of gigabytes: it is converted ahead
        // into a buffer of its own (`converted`) and moves into its stage as the microkernel plans.
        // Decoded by its stage instead, Devstral Small 2's 128k prompt over an `e4m3` cache took
        // 24.1 s to its first token on an RTX PRO 6000, against 22.1 s converted ahead, under the
        // same tuned routine.
        let words = |operand: AttentionOperand| match problem.elements.get(operand) {
            stored @ ElemType::Float(kind)
                if read.get(operand) != stored
                    && (operand != AttentionOperand::Query
                        || stored.size() == u32::BITS as usize / 8) =>
            {
                Some(kind)
            }
            _ => None,
        };
        let delivery = |operand: AttentionOperand, planned: Delivery| match words(operand) {
            Some(_) => Delivery::SyncPerUnit,
            None => planned,
        };
        let arg = |tensor: &PlainTensor,
                   dimensions: &'static [Dimension],
                   width: usize,
                   element: ElemType| {
            launcher
                .arg(tensor.binding.clone(), element)
                .dimensions(dimensions)
                .vectorize(width)
        };
        let read_operand =
            |tensor: &PlainTensor,
             dimensions: &'static [Dimension],
             width: usize,
             operand: AttentionOperand| match words(operand) {
                Some(kind) => {
                    arg(tensor, dimensions, width, ElemType::UInt(UIntKind::U32)).packed(kind)
                }
                None => arg(tensor, dimensions, width, read.get(operand)),
            };
        let query_delivery = delivery(
            AttentionOperand::Query,
            blueprint
                .microkernel
                .query_delivery(&widths, &problem.work_item()),
        );
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
                .or_broken(ADMITTED_PLAN_RUNS),
            None => Ok(bound),
        };
        let planned = blueprint
            .microkernel
            .delivery(&widths, &problem.work_item());
        // A side of a float cache bound under the launcher as its microkernel plans its moves.
        let read_side = |tensor: &PlainTensor,
                         dimensions: &'static [Dimension],
                         width: usize,
                         operand: AttentionOperand|
         -> Result<TileTensorLaunch, Decline> {
            Ok(read_operand(tensor, dimensions, width, operand)
                .delivery(delivery(operand, planned))
                .build()
                .or_broken(ADMITTED_PLAN_RUNS)?
                .arg())
        };
        let cache = match &self.kv.cache {
            AttentionCache::Float { key, value } => {
                self.bind_float_cache(key, value, blueprint, &read_side)?
            }
            AttentionCache::Packed(cache) => {
                AttentionCacheBinding::Packed(cache.bind(&launcher, self.kv.pages.as_ref())?)
            }
        };
        let out = runs(
            arg(
                &self.out_view,
                &[B, KV, G, Q, DV],
                widths.out,
                problem.elements.out,
            )
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?
            .arg(),
        )?;
        // A value a query, written by whoever holds its row.
        let logsumexp = self
            .logsumexp
            .as_ref()
            .map(|(_, view)| {
                arg(view, &[B, KV, G, Q], 1, AttentionRoutineProblem::SUM)
                    .build()
                    .or_broken(ADMITTED_PLAN_RUNS)
                    .map(Bound::arg)
            })
            .transpose()?;
        // A cell a read: a mask broadcast over a dim lies there at stride zero.
        let mask = self
            .mask
            .as_ref()
            .map(|mask| {
                arg(mask, &[B, KV, G, Q, S], 1, mask.elem)
                    .build()
                    .or_broken(ADMITTED_PLAN_RUNS)
                    .map(Bound::arg)
            })
            .transpose()?;
        // The query last: one read at another element is converted into a buffer of its own, the
        // one launch a binding makes.
        let query_read = read.get(AttentionOperand::Query);
        let query = match query_read != problem.elements.get(AttentionOperand::Query)
            && words(AttentionOperand::Query).is_none()
        {
            true => converted(&self.client, &self.query, query_read),
            false => self.query.clone(),
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
            .or_broken(ADMITTED_PLAN_RUNS)?
            .arg(),
        )?;
        Ok(AttentionBinding {
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
            query_count: self.query_count() as u32,
        })
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
    /// keeps one for every box, otherwise filled for this launch.
    pub fn box_arrivals(&self, box_queries: usize) -> PlainTensor {
        let problem = self.problem();
        let boxes = problem.batch
            * problem.num_kv_heads
            * problem.group
            * problem.seq_q.div_ceil(box_queries);
        let covers = |kept: &&PlainTensor| {
            kept.elem == Self::COUNTER && kept.shape().num_elements() >= boxes
        };
        self.arrivals
            .as_ref()
            .filter(covers)
            .cloned()
            .unwrap_or_else(|| {
                PlainTensor::of_handle(TensorHandle::zeros(&self.client, [boxes], Self::COUNTER))
            })
    }
}

impl AttentionTensors {
    /// The elements of float operands, where the routines take them and `client`'s device runs
    /// them: the query in an element the device computes in, and the keys and the values each in
    /// one of those or in `e4m3` ([`VERIFIED_CACHE`](Self::VERIFIED_CACHE)), whatever the other
    /// side's, where the device stores it. A routine reads each side as stored or converted as it
    /// stages it
    /// ([`AttentionWorkItem::readings`](cubek_microkernel::attention::AttentionWorkItem::readings)),
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
    ) -> Result<(), Decline> {
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
    fn admit_element(client: &Client, query: &PlainTensor) -> Result<(), Decline> {
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
    fn group(query: &PlainTensor, kv_heads: usize) -> Result<usize, Decline> {
        let heads = query.shape()[1];
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(ATTENTION_GROUP_UNEVEN);
        }
        Ok(heads / kv_heads)
    }

    /// The tensors of an attention of `query`, `group` heads a key head, over `kv`: the output
    /// allocated, and the query and the output viewed along the query dimension. Where `query_runs`
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
        let (dimension, out, out_plain) = match &query_runs {
            None => {
                let (out, out_plain) = PlainTensor::empty_contiguous(
                    client,
                    [batch, queries, heads, value].into(),
                    out_elem,
                );
                (
                    QueryDimension::new(queries, scoring.visibility),
                    out,
                    out_plain,
                )
            }
            Some(_) => {
                let (out, end_to_end) =
                    PlainTensor::empty_contiguous(client, [queries, heads, value].into(), out_elem);
                let strides = &end_to_end.binding.strides;
                let out_plain = end_to_end.viewed(Metadata::new(
                    [batch, queries, heads, value],
                    [0, strides[0], strides[1], strides[2]],
                ));
                (QueryDimension::Positions, out, out_plain)
            }
        };
        Self {
            client: client.clone(),
            query: dimension.view(&query, group, [0, 1, 2, 3]),
            out_view: dimension.view(&out_plain, group, [0, 2, 1, 3]),
            out,
            kv,
            lengths,
            query_runs,
            scoring,
            dimension,
            arrivals: None,
            logsumexp: None,
            mask: None,
        }
    }

    /// `lengths` as one `u32` value per sequence, `(sequences,)`, read by stride: a `(1,)`
    /// tensor serves every sequence as a stride-0 view of its one element rather than a copy, so
    /// the kernel still reads the buffer an advance moves in place.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_LENGTHS_UNREAD`].
    fn one_per_sequence(lengths: PlainTensor, sequences: usize) -> Result<PlainTensor, Decline> {
        if lengths.elem != Self::COUNTER {
            return Err(ATTENTION_LENGTHS_UNREAD);
        }
        match lengths.shape().as_slice() {
            [1] => Ok(lengths.viewed(Metadata::new([sequences], [0]))),
            [lengths_count] if *lengths_count == sequences => Ok(lengths),
            _ => Err(ATTENTION_LENGTHS_UNREAD),
        }
    }

    /// The queries a sequence has, or, where they lie end to end, all of them.
    fn query_count(&self) -> usize {
        match &self.query_runs {
            Some(_) => self.out.shape()[0],
            None => self.problem().seq_q,
        }
    }

    /// The float `key` and `value` bound as `blueprint` states them, each side read by `read_side`
    /// over the dimensions its buffer steps, or with its keys innermost where its microkernel reads
    /// it so, under its per-tensor scale where it has one, and through the page table where the
    /// cache lies in pages ([`page_table`]).
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    fn bind_float_cache<M: AttentionMicrokernel, SM>(
        &self,
        key: &PlainTensor,
        value: &PlainTensor,
        blueprint: &AttentionBlueprint<M, SM>,
        read_side: &impl Fn(
            &PlainTensor,
            &'static [Dimension],
            usize,
            AttentionOperand,
        ) -> Result<TileTensorLaunch, Decline>,
    ) -> Result<AttentionCacheBinding, Decline> {
        // A side's scale is one value over the whole side, read at no coordinate: bound over the
        // side's head dim alone, which no page moves, it broadcasts to every value, and the
        // microkernel applies it to the sums the side enters.
        let per_tensor = |scale: &PlainTensor, head: Dimension| {
            TileTensorLaunch::new(
                scale.binding.clone().into_tensor_arg(),
                scale.elem,
                1,
                TileSpec::new(Projection::new(&[head], &[PhysicalAxisMap::broadcast()])),
            )
        };
        let order = blueprint.microkernel.cache_order();
        // The side's scale is stated before its table, so the table places the values under it.
        let bind = |tensor: &PlainTensor,
                    stored: &'static [Dimension],
                    along_keys: &'static [Dimension],
                    width: usize,
                    operand: AttentionOperand,
                    scale: Option<&PlainTensor>,
                    head: Dimension| {
            let arg = match operand.lines(order) {
                CacheLines::AsStored => read_side(tensor, stored, width, operand)?,
                CacheLines::AlongKeys => {
                    read_side(&tensor.transposed(), along_keys, width, operand)?
                }
            };
            let arg = match scale {
                Some(scale) => arg.scales(per_tensor(scale, head)),
                None => arg,
            };
            match &self.kv.pages {
                Some(pages) => arg
                    .indexed_covering(page_table(pages), Indexing::Coordinate(B))
                    .or_broken(ADMITTED_PLAN_RUNS),
                None => Ok(arg),
            }
        };
        let (widths, scales) = (&blueprint.widths, &self.kv.scales);
        let key = bind(
            key,
            &[B, KV, S, D],
            &[B, KV, D, S],
            widths.key,
            AttentionOperand::Key,
            scales.key.as_ref(),
            D,
        )?;
        let value = bind(
            value,
            &[B, KV, S, DV],
            &[B, KV, DV, S],
            widths.value,
            AttentionOperand::Value,
            scales.value.as_ref(),
            DV,
        )?;
        Ok(AttentionCacheBinding::Tiles { key, value })
    }

    /// Every sequence's queries end to end, `query` `[Σ q_b, h, d]`, viewed as a query of each
    /// sequence `ends` counts, `[b, h, Σ q_b, d]`, the batch at stride zero: each sequence's run of
    /// it is placed by its entry of `ends` as the launch binds it.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_QUERIES_UNREAD`].
    fn end_to_end(query: PlainTensor, ends: &PlainTensor) -> Result<PlainTensor, Decline> {
        let (shape, strides) = (&query.binding.shape, &query.binding.strides);
        if shape.len() != 3 || ends.elem != Self::COUNTER || ends.shape().rank() != 1 {
            return Err(ATTENTION_QUERIES_UNREAD);
        }
        Ok(query.viewed(Metadata::new(
            [ends.shape()[0], shape[1], shape[0], shape[2]],
            [0, strides[1], strides[0], strides[2]],
        )))
    }

    /// Float operands of one element the routines take and `client`'s device runs, the keys' head
    /// dim the query's.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`], [`ELEMENT_NOT_ON_DEVICE`] and
    /// [`ATTENTION_QUERIES_UNREAD`].
    fn admit_float(
        client: &Client,
        query: &PlainTensor,
        key: &PlainTensor,
        value: &PlainTensor,
    ) -> Result<(), Decline> {
        Self::admit_elements(client, query, key, value)?;
        if key.shape()[3] != query.shape()[3] {
            return Err(ATTENTION_QUERIES_UNREAD);
        }
        Ok(())
    }

    /// `cache`, a packed one under `query` of an element the routines take and `client`'s device
    /// runs, the query as wide as the keys' lines.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`], [`ELEMENT_NOT_ON_DEVICE`] and
    /// [`ATTENTION_QUERIES_UNREAD`].
    fn admit_packed(
        client: &Client,
        query: &PlainTensor,
        cache: &PackedAttentionCache,
    ) -> Result<(), Decline> {
        Self::admit_element(client, query)?;
        let coords = cache.coords();
        let widened = match cache.fixup {
            Some(_) => 2 * coords,
            None => coords,
        };
        if query.shape()[3] != widened {
            return Err(ATTENTION_QUERIES_UNREAD);
        }
        Ok(())
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

impl AttentionSplitSlots {
    /// The outputs, the maxes and the sums, as a kernel's launch arguments.
    pub fn to_args(&self) -> [TensorArg; 3] {
        [&self.out, &self.max, &self.sum].map(|slot| slot.clone().binding().into_tensor_arg())
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::*;
    use core::num::NonZeroUsize;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::tile::launch::Decline;
    use cubecl::zspace::Shape;
    use cubecl::zspace::metadata::Metadata;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_test_utils::TestInput;

    /// The shapes of a packed test cache: its keys' words and norms `[rows, 2, 4, 1]`, its values'
    /// words and norms their own shapes, and the page table where the cache lies in pages.
    struct PackedShapes<'a> {
        pub rows: usize,
        pub value_words: &'a [usize],
        pub value_norms: &'a [usize],
        pub pages: Option<[usize; 2]>,
    }

    /// The attention of one sequence's three queries of four heads, of head dim eight, over a
    /// packed cache of four-bit keys and values of `shapes`.
    fn attend_packed(shapes: PackedShapes) -> Option<Decline> {
        let client: Client = cubecl::test_device().client();
        let tensor = |shape: &[usize], element: ElemType| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(element)
                    .zeros()
                    .generate_without_host_data(),
            )
        };
        let (f32, u32) = (f32::elem_type_native(), u32::elem_type_native());
        let operand = |words: &[usize], norms: &[usize]| PackedAttentionOperand {
            words: tensor(words, u32),
            norms: tensor(norms, f32),
            codebook: tensor(&[16], f32),
        };
        let rows = shapes.rows;
        let cache = PackedAttentionCache::new(
            operand(&[rows, 2, 4, 1], &[rows, 2, 4, 1]),
            operand(shapes.value_words, shapes.value_norms),
            KvBitWidths { key: 4, value: 4 },
            None,
        );
        let kv = KvSource::new(AttentionCache::Packed(cache));
        // A table of no column is a view of one cell, which no buffer holds empty.
        let kv = match shapes.pages {
            Some(shape) => kv.paged(tensor(&[1, 1], u32).viewed(Metadata::new(shape, [1, 1]))),
            None => kv,
        };
        let queries = AttentionQueries::Dense {
            query: tensor(&[1, 4, 3, 8], f32),
            lengths: tensor(&[1], u32),
        };
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);
        AttentionTensors::attention(&client, queries, kv, scoring).err()
    }

    /// Packed buffers that disagree on their positions state no cache.
    #[test]
    fn packed_buffers_of_other_positions_decline() {
        let declined = attend_packed(PackedShapes {
            rows: 1,
            value_words: &[1, 2, 8, 1],
            value_norms: &[1, 2, 8, 1],
            pages: None,
        });

        assert_eq!(declined, Some(PACKED_CACHE_UNREAD));
    }

    /// A packed operand's norms of another rank than its words' state no cache.
    #[test]
    fn packed_norms_not_of_rank_four_decline() {
        let declined = attend_packed(PackedShapes {
            rows: 1,
            value_words: &[1, 2, 4, 1],
            value_norms: &[1, 2, 4],
            pages: None,
        });

        assert_eq!(declined, Some(PACKED_CACHE_UNREAD));
    }

    /// A packed cache laid out as sequences holds one row a sequence of the query.
    #[test]
    fn packed_sequences_of_another_batch_than_the_query_decline() {
        let declined = attend_packed(PackedShapes {
            rows: 2,
            value_words: &[2, 2, 4, 1],
            value_norms: &[2, 2, 4, 1],
            pages: None,
        });

        assert_eq!(declined, Some(CACHE_SEQUENCES_DISAGREE));
    }

    /// A packed cache in pages is read through a table whose rows name pages.
    #[test]
    fn packed_pages_under_an_empty_table_row_decline() {
        let declined = attend_packed(PackedShapes {
            rows: 2,
            value_words: &[2, 2, 4, 1],
            value_norms: &[2, 2, 4, 1],
            pages: Some([1, 0]),
        });

        assert_eq!(declined, Some(PAGE_TABLE_UNREAD));
    }

    /// A tensor of `shape` and `element`, zeros.
    fn zeros(shape: &[usize], element: ElemType) -> PlainTensor {
        let client: Client = cubecl::test_device().client();
        PlainTensor::of_handle(
            TestInput::builder(client, Shape::from(shape.to_vec()))
                .dtype(element)
                .zeros()
                .generate_without_host_data(),
        )
    }

    /// The attention of one sequence's three queries of four heads over four keys of two heads, of
    /// head dim eight, float, attending `lengths`.
    fn attend(
        query: &[usize],
        value: &[usize],
        lengths: PlainTensor,
    ) -> Result<AttentionTensors, Decline> {
        let client: Client = cubecl::test_device().client();
        let f32 = f32::elem_type_native();
        AttentionTensors::new(
            &client,
            zeros(query, f32),
            zeros(&[1, 2, 4, 8], f32),
            zeros(value, f32),
            lengths,
            AttentionScoring::standard(AttentionVisibility::Attended, 8),
        )
    }

    const QUERY: [usize; 4] = [1, 4, 3, 8];
    const VALUE: [usize; 4] = [1, 2, 4, 8];

    fn one_length() -> PlainTensor {
        zeros(&[1], u32::elem_type_native())
    }

    #[test]
    fn a_query_not_of_rank_four_declines() {
        let declined = attend(&[4, 3, 8], &VALUE, one_length()).err();

        assert_eq!(declined, Some(ATTENTION_QUERIES_UNREAD));
    }

    #[test]
    fn a_query_of_another_head_dim_declines() {
        let declined = attend(&[1, 4, 3, 4], &VALUE, one_length()).err();

        assert_eq!(declined, Some(ATTENTION_QUERIES_UNREAD));
    }

    #[test]
    fn values_of_other_positions_than_the_keys_decline() {
        let declined = attend(&QUERY, &[1, 2, 5, 8], one_length()).err();

        assert_eq!(declined, Some(CACHE_SHAPES_DISAGREE));
    }

    #[test]
    fn a_query_of_another_batch_declines() {
        let declined = attend(&[2, 4, 3, 8], &VALUE, one_length()).err();

        assert_eq!(declined, Some(CACHE_SEQUENCES_DISAGREE));
    }

    #[test]
    fn float_lengths_decline() {
        let lengths = zeros(&[1], f32::elem_type_native());

        let declined = attend(&QUERY, &VALUE, lengths).err();

        assert_eq!(declined, Some(ATTENTION_LENGTHS_UNREAD));
    }

    #[test]
    fn lengths_of_other_sequences_decline() {
        let lengths = zeros(&[3], u32::elem_type_native());

        let declined = attend(&QUERY, &VALUE, lengths).err();

        assert_eq!(declined, Some(ATTENTION_LENGTHS_UNREAD));
    }

    #[test]
    fn a_mask_of_another_shape_than_the_scores_declines() {
        let tensors = attend(&QUERY, &VALUE, one_length()).expect("an attention");
        let mask = zeros(&[1, 4, 4, 3], u32::elem_type_native());

        let declined = tensors.with_mask(mask).err();

        assert_eq!(declined, Some(ATTENTION_MASK_UNREAD));
    }

    /// Two sequences' two queries end to end, over their float keys and values.
    fn attend_packed_queries(ends: PlainTensor) -> Result<AttentionTensors, Decline> {
        let client: Client = cubecl::test_device().client();
        let f32 = f32::elem_type_native();
        let queries = AttentionQueries::Packed {
            query: zeros(&[2, 4, 8], f32),
            ends,
            longest: 1,
            lengths: one_length(),
        };
        let kv = KvSource::new(AttentionCache::Float {
            key: zeros(&[2, 2, 4, 8], f32),
            value: zeros(&[2, 2, 4, 8], f32),
        });
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);
        AttentionTensors::attention(&client, queries, kv, scoring)
    }

    #[test]
    fn float_ends_of_packed_queries_decline() {
        let ends = zeros(&[2], f32::elem_type_native());

        let declined = attend_packed_queries(ends).err();

        assert_eq!(declined, Some(ATTENTION_QUERIES_UNREAD));
    }

    #[test]
    fn packed_queries_take_no_mask() {
        let ends = zeros(&[2], u32::elem_type_native());
        let tensors = attend_packed_queries(ends).expect("an attention");

        let declined = tensors.with_mask(one_length()).err();

        assert_eq!(declined, Some(ATTENTION_MASK_UNREAD));
    }

    #[test]
    fn packed_queries_write_no_logsumexp() {
        let ends = zeros(&[2], u32::elem_type_native());
        let tensors = attend_packed_queries(ends).expect("an attention");

        let declined = tensors.with_logsumexp().err();

        assert_eq!(declined, Some(ATTENTION_LOGSUMEXP_UNWRITTEN));
    }

    /// Kept counters fewer than the launch's twelve boxes, or not counters, are left for counters
    /// of the launch's own.
    #[test]
    fn arrivals_that_do_not_cover_the_boxes_are_left() {
        let (u32, f32) = (u32::elem_type_native(), f32::elem_type_native());
        for kept in [zeros(&[11], u32), zeros(&[12], f32)] {
            let tensors = attend(&QUERY, &VALUE, one_length()).expect("an attention");

            let arrivals = tensors.with_arrivals(Some(kept)).box_arrivals(1);

            assert_eq!((arrivals.shape().num_elements(), arrivals.elem), (12, u32));
        }
    }

    /// A cache's scale is one `f32`.
    #[test]
    fn a_scale_per_key_head_declines() {
        let client: Client = cubecl::test_device().client();
        let f32 = f32::elem_type_native();
        let kv = KvSource::new(AttentionCache::Float {
            key: zeros(&[1, 2, 4, 8], f32),
            value: zeros(&VALUE, f32),
        })
        .scaled(CacheScales {
            key: Some(zeros(&[2], f32)),
            value: None,
        });
        let queries = AttentionQueries::Dense {
            query: zeros(&QUERY, f32),
            lengths: one_length(),
        };
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);

        let declined = AttentionTensors::attention(&client, queries, kv, scoring).err();

        assert_eq!(declined, Some(CACHE_SCALE_UNREAD));
    }

    /// A value of fewer coordinates than a key's leaves the output's last ones unread.
    #[test]
    fn packed_values_of_fewer_coordinates_decline() {
        let declined = attend_packed(PackedShapes {
            rows: 1,
            value_words: &[1, 2, 4, 0],
            value_norms: &[1, 2, 4, 1],
            pages: None,
        });

        assert_eq!(declined, Some(PACKED_CACHE_UNREAD));
    }

    /// A four-bit packed cache of one sequence's four keys of two heads, its value's words and norms
    /// as `break_value` leaves them.
    fn packed_cache(break_value: impl FnOnce(&mut PackedAttentionOperand)) -> PackedAttentionCache {
        let (f32, u32) = (f32::elem_type_native(), u32::elem_type_native());
        let operand = || PackedAttentionOperand {
            words: zeros(&[1, 2, 4, 1], u32),
            norms: zeros(&[1, 2, 4, 1], f32),
            codebook: zeros(&[16], f32),
        };
        let mut value = operand();
        break_value(&mut value);
        let bits = KvBitWidths { key: 4, value: 4 };
        PackedAttentionCache::new(operand(), value, bits, None)
    }

    /// The attention of one sequence's queries over `kv`.
    fn attend_over(kv: KvSource) -> Result<AttentionTensors, Decline> {
        let client: Client = cubecl::test_device().client();
        let queries = AttentionQueries::Dense {
            query: zeros(&QUERY, f32::elem_type_native()),
            lengths: one_length(),
        };
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);
        AttentionTensors::attention(&client, queries, kv, scoring)
    }

    fn declined_packed(cache: PackedAttentionCache) -> Option<Decline> {
        attend_over(KvSource::new(AttentionCache::Packed(cache))).err()
    }

    fn float_kv() -> KvSource {
        let f32 = f32::elem_type_native();
        KvSource::new(AttentionCache::Float {
            key: zeros(&VALUE, f32),
            value: zeros(&VALUE, f32),
        })
    }

    #[test]
    fn packed_words_not_u32_decline() {
        let cache =
            packed_cache(|value| value.words = zeros(&[1, 2, 4, 1], f32::elem_type_native()));

        assert_eq!(declined_packed(cache), Some(PACKED_CACHE_UNREAD));
    }

    #[test]
    fn packed_fields_of_a_row_apart_decline() {
        let cache = packed_cache(|value| {
            let shape = [1, 2, 4, 1];
            value.words =
                zeros(&shape, u32::elem_type_native()).viewed(Metadata::new(shape, [8, 4, 1, 2]));
        });

        assert_eq!(declined_packed(cache), Some(PACKED_CACHE_UNREAD));
    }

    #[test]
    fn packed_norms_not_f32_decline() {
        let cache =
            packed_cache(|value| value.norms = zeros(&[1, 2, 4, 1], u32::elem_type_native()));

        assert_eq!(declined_packed(cache), Some(PACKED_CACHE_UNREAD));
    }

    #[test]
    fn a_codebook_short_of_its_indices_declines() {
        let cache = packed_cache(|value| value.codebook = zeros(&[15], f32::elem_type_native()));

        assert_eq!(declined_packed(cache), Some(PACKED_CACHE_UNREAD));
    }

    #[test]
    fn a_packed_width_of_three_bits_declines() {
        let cache = PackedAttentionCache {
            bits: KvBitWidths { key: 3, value: 4 },
            ..packed_cache(|_| {})
        };

        assert_eq!(declined_packed(cache), Some(PACKED_WIDTH_UNREAD));
    }

    #[test]
    fn packed_queries_of_another_head_dim_decline() {
        let client: Client = cubecl::test_device().client();
        let queries = AttentionQueries::Dense {
            query: zeros(&[1, 4, 3, 16], f32::elem_type_native()),
            lengths: one_length(),
        };
        let kv = KvSource::new(AttentionCache::Packed(packed_cache(|_| {})));
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 16);

        let declined = AttentionTensors::attention(&client, queries, kv, scoring).err();

        assert_eq!(declined, Some(ATTENTION_QUERIES_UNREAD));
    }

    #[test]
    fn a_packed_caches_logsumexp_declines() {
        let kv = KvSource::new(AttentionCache::Packed(packed_cache(|_| {})));
        let tensors = attend_over(kv).expect("an attention");

        let declined = tensors.with_logsumexp().err();

        assert_eq!(declined, Some(ATTENTION_LOGSUMEXP_UNWRITTEN));
    }

    /// Queries of `2^32` elements, views of one: a head dim at stride zero counts each.
    #[test]
    fn queries_past_a_u32_decline() {
        let query = zeros(&[1], f32::elem_type_native())
            .viewed(Metadata::new([1, 4, 1 << 27, 8], [1, 1, 1, 1]));
        let client: Client = cubecl::test_device().client();
        let queries = AttentionQueries::Dense {
            query,
            lengths: one_length(),
        };
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);

        let declined = AttentionTensors::attention(&client, queries, float_kv(), scoring).err();

        assert_eq!(declined, Some(ATTENTION_QUERIES_UNCOUNTED));
    }

    #[test]
    fn a_float_page_table_declines() {
        let kv = float_kv().paged(zeros(&[1, 1], f32::elem_type_native()));

        assert_eq!(attend_over(kv).err(), Some(PAGE_TABLE_UNREAD));
    }

    #[test]
    fn a_page_table_of_other_sequences_declines() {
        let kv = float_kv().paged(zeros(&[2, 1], u32::elem_type_native()));

        assert_eq!(attend_over(kv).err(), Some(PAGE_TABLE_UNREAD));
    }

    #[test]
    fn a_block_table_of_other_sequences_declines() {
        let blocks = zeros(&[2, 1], u32::elem_type_native());
        let kv = float_kv().indexed(blocks, NonZeroUsize::MIN);

        assert_eq!(attend_over(kv).err(), Some(PAGE_TABLE_UNREAD));
    }

    /// A key head's positions at stride zero have no block to step.
    #[test]
    fn listed_blocks_of_repeated_positions_decline() {
        let f32 = f32::elem_type_native();
        let repeated = || {
            let strides = [16, 8, 0, 1];
            zeros(&[1, 2, 1, 8], f32).viewed(Metadata::new(VALUE, strides))
        };
        let kv = KvSource::new(AttentionCache::Float {
            key: repeated(),
            value: repeated(),
        })
        .indexed(zeros(&[1, 1], u32::elem_type_native()), NonZeroUsize::MIN);

        assert_eq!(attend_over(kv).err(), Some(KEY_BLOCKS_UNPLACED));
    }
}
