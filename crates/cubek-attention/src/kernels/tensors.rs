//! The tensors of one attention, and how a launch binds them.

use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionCacheBinding, AttentionCacheLayout,
    AttentionRoutineProblem, AttentionScoring, PackedAttentionCache, QueryAxis,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::{ComptimeOptionArgs, TensorArg};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Launcher, launch::Grid as LaunchGrid};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperands, B, D, DV, G, KV, Q, S,
};
use cubek_microkernel::matmul::MatmulOrder;
use cubek_std::routine::ADMITTED_PLAN_RUNS;
use cubek_std::routine::Grid;

/// What a launch declines before preparing anything: an element the routines do not take, or
/// operands of two elements.
pub const ATTENTION_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel attention takes f32, f16 and bf16 operands of one element",
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
    cache: AttentionCache,
    /// The `u32` attended length of each sequence, `[b]`: on the device, so a captured launch
    /// replays as the cache grows.
    lengths: PlainTensor,
    /// The output, `[b, q, h, dv]`.
    out: TensorHandle,
    /// The output as the kernel writes it, `[b, kv, g, q, dv]` over `out`'s buffer.
    out_view: PlainTensor,
    scoring: AttentionScoring,
    axis: QueryAxis,
    /// The `u32` counters the cubes of a box count themselves in at, held by the caller across
    /// launches ([`with_arrivals`](Self::with_arrivals)); `None` where each launch makes its own.
    arrivals: Option<PlainTensor>,
}

impl core::fmt::Debug for AttentionTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the attention.
        f.debug_struct("AttentionTensors")
            .field("query", &self.query)
            .field("cache", &self.cache)
            .field("lengths", &self.lengths)
            .field("out", &self.out)
            .field("out_view", &self.out_view)
            .field("scoring", &self.scoring)
            .field("axis", &self.axis)
            .field("arrivals", &self.arrivals)
            .finish()
    }
}

/// The keys and values an attention reads.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, held by value as the launch's other operands are; both arms are tensors"
)]
pub enum AttentionCache {
    /// Float keys `[b, kv, s, d]` and values `[b, kv, s, dv]`; or, read through `pages`, the `u32`
    /// `[b, pages_per_sequence]` table naming each sequence's pages in order, two pools of pages,
    /// `[pages, kv, page_size, d]` and `[pages, kv, page_size, dv]`.
    Float {
        key: PlainTensor,
        value: PlainTensor,
        pages: Option<PlainTensor>,
    },
    /// A packed cache, decoded as it is read.
    Packed(PackedAttentionCache),
}

impl AttentionTensors {
    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of `key`
    /// (`[b, kv, s, d]`) and `value` (`[b, kv, s, dv]`), the queries the last `q` positions of the
    /// attended keys, each scoring the keys as `scoring` says; the output `[b, q, h, dv]`
    /// allocated.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`] and [`ATTENTION_GROUP_UNEVEN`]: what the routines
    /// decline whatever the device, and [`ELEMENT_NOT_ON_DEVICE`]: an element this device's
    /// runtime does not run.
    ///
    /// # Panics
    ///
    /// Operands not of rank four, or whose batch, key or head dims do not agree: no attention at
    /// all.
    pub fn new(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        Self::admit_float(client, &query, &key, &value)?;
        let (q_shape, k_shape, v_shape) = (query.shape(), key.shape(), value.shape());
        assert!(
            q_shape.num_dims() == 4 && k_shape.num_dims() == 4 && v_shape.num_dims() == 4,
            "attention's operands are [b, h, q, d], [b, kv, s, d] and [b, kv, s, dv]"
        );
        let batch = q_shape[0];
        assert!(
            k_shape[0] == batch && v_shape[0] == batch && v_shape[1] == k_shape[1],
            "the operands' batch and key head dims agree"
        );
        assert!(
            k_shape[3] == q_shape[3] && v_shape[2] == k_shape[2],
            "the keys' head dim is the query's, and the values hold a row per key"
        );
        let group = Self::group(&query, k_shape[1])?;
        let cache = AttentionCache::Float {
            key,
            value,
            pages: None,
        };
        Ok(Self::assemble(
            client, query, group, cache, lengths, scoring,
        ))
    }

    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of the keys
    /// and values `pages` lays out in two pools of pages, `key` (`[pages, kv, page_size, d]`) and
    /// `value` (`[pages, kv, page_size, dv]`): sequence `b`'s position `s` lies in page
    /// `pages[b, s / page_size]`, `pages` a `u32` `[b, pages_per_sequence]`. Each sequence holds
    /// as many keys as its row of the table has pages; otherwise as [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), a table that is not `u32` of rank two with one row per sequence,
    /// pools that disagree on their pages, key heads or page size, or an empty pool, page or
    /// table row.
    pub fn paged(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        pages: PlainTensor,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        Self::admit_float(client, &query, &key, &value)?;
        let (q_shape, k_shape, v_shape) = (query.shape(), key.shape(), value.shape());
        assert!(
            q_shape.num_dims() == 4 && k_shape.num_dims() == 4 && v_shape.num_dims() == 4,
            "paged attention's operands are [b, h, q, d], [pages, kv, page_size, d] and \
             [pages, kv, page_size, dv]"
        );
        let pages = Self::page_table(pages, q_shape[0]);
        let pages_per_sequence = pages.shape()[1];
        let page_size = k_shape[2];
        assert!(
            v_shape[0] == k_shape[0] && v_shape[1] == k_shape[1] && v_shape[2] == page_size,
            "the pools agree on their pages, key heads and page size"
        );
        assert!(
            k_shape[0] > 0 && page_size > 0 && pages_per_sequence > 0,
            "a paged attention reads a pool, its pages and its table rows non-empty"
        );
        assert_eq!(k_shape[3], q_shape[3], "the keys' head dim is the query's");
        let group = Self::group(&query, k_shape[1])?;
        let cache = AttentionCache::Float {
            key,
            value,
            pages: Some(pages),
        };
        Ok(Self::assemble(
            client, query, group, cache, lengths, scoring,
        ))
    }

    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of a packed
    /// cache, the queries the last `q` positions of the attended keys, each scoring the keys as
    /// `scoring` says; the output `[b, q, h, dv]` allocated, `dv` the cache's head dim.
    ///
    /// Where the keys carry a QJL fix-up, `query` is the query followed along its head dim by its
    /// projection, `d` twice the cache's head dim: the keys' lines run as far, their second half
    /// the fix-up ([`PackedLines`](cubek_microkernel::attention::PackedLines)).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new) for the query's element and the group, and
    /// [`PACKED_WIDTH_UNREAD`].
    ///
    /// # Panics
    ///
    /// A cache [`PackedAttentionCache::new`] refuses, or a query whose head dim is not the
    /// cache's (twice it with a fix-up).
    pub fn packed(
        client: &Client,
        query: PlainTensor,
        cache: PackedAttentionCache,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        Self::admit_element(client, &query)?;
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
        let group = Self::group(&query, cache.key.words.shape()[1])?;
        let cache = AttentionCache::Packed(cache);
        Ok(Self::assemble(
            client, query, group, cache, lengths, scoring,
        ))
    }

    /// The elements the routines are verified at.
    pub const VERIFIED: [ElemType; 3] = [
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
    ];

    /// Float operands of one element the routines take and `client`'s device runs.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_ELEMENT_UNVERIFIED`] and [`ELEMENT_NOT_ON_DEVICE`].
    fn admit_float(
        client: &Client,
        query: &PlainTensor,
        key: &PlainTensor,
        value: &PlainTensor,
    ) -> Result<(), Refusal> {
        if query.elem != key.elem || query.elem != value.elem {
            return Err(ATTENTION_ELEMENT_UNVERIFIED);
        }
        Self::admit_element(client, query)
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

    /// The tensors of an attention of `query`, `group` heads a key head, over `cache`: the output
    /// allocated, and the query and the output viewed along the query axis.
    fn assemble(
        client: &Client,
        query: PlainTensor,
        group: usize,
        cache: AttentionCache,
        lengths: PlainTensor,
        scoring: AttentionScoring,
    ) -> Self {
        let q_shape = query.shape();
        let (batch, heads, queries) = (q_shape[0], q_shape[1], q_shape[2]);
        let axis = QueryAxis::new(queries);
        // Dense, as every arm of the attention seam allocates it: a token's heads are one row, so
        // the caller merges them by a reshape. The views below are a factorization of these
        // strides, which a pitched allocation would invalidate.
        let (out, out_plain) = PlainTensor::empty_contiguous(
            client,
            [batch, queries, heads, cache.val_dim()].into(),
            query.elem,
        );
        Self {
            client: client.clone(),
            query: axis.view(&query, group, [0, 1, 2, 3]),
            out_view: axis.view(&out_plain, group, [0, 2, 1, 3]),
            out,
            cache,
            lengths: Self::one_per_sequence(lengths, batch),
            scoring,
            axis,
            arrivals: None,
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

    /// `pages` as the page table of `sequences`: `u32` `(sequences, pages_per_sequence)`, read by
    /// stride, so it is not copied.
    ///
    /// # Panics
    ///
    /// If it is not `u32`, not rank 2, or holds another count of rows.
    fn page_table(pages: PlainTensor, sequences: usize) -> PlainTensor {
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
        pages
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

    /// The problem the tensors state.
    pub fn problem(&self) -> AttentionRoutineProblem {
        let [batch, queries, heads, value] = self.out.shape().dims::<4>();
        let kv_heads = self.cache.num_kv_heads();
        let (group, queries, visibility) =
            self.axis
                .read(heads / kv_heads, queries, self.scoring.visibility);
        let element = self.query.elem;
        let head = self.query.shape()[4];
        let start = PlainTensor::start_bytes;
        let (key_alignment, value_alignment, key_start, value_start) = match &self.cache {
            AttentionCache::Float { key, value, .. } => (
                key.elem_alignment(),
                value.elem_alignment(),
                start(key),
                start(value),
            ),
            // The decoded lines lie as dense rows of the head dim.
            AttentionCache::Packed(_) => (
                ElemAlignment::dense_rows(head),
                ElemAlignment::dense_rows(value),
                0,
                0,
            ),
        };
        AttentionRoutineProblem {
            batch,
            num_kv_heads: kv_heads,
            group,
            seq_q: queries,
            seq_kv: self.cache.seq_kv(),
            head_dim: head,
            val_dim: value,
            visibility,
            elements: AttentionOperands {
                query: element,
                key: element,
                value: element,
                out: element,
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
            cache: self.cache.layout(),
        }
    }

    /// The cache the keys and values are read from.
    pub fn cache(&self) -> &AttentionCache {
        &self.cache
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

    /// The output, `[b, q, h, dv]`.
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
            partitioning.as_cubek().space(),
            LaunchGrid::Stated {
                cube_count: grid.launch.cube_count.clone(),
                cube_dim: grid.launch.cube_dim,
            },
        )
        .expect(ADMITTED_PLAN_RUNS);
        let widths = blueprint.widths;
        let problem = self.problem();
        let stored = problem.elements.query;
        let read = blueprint
            .microkernel
            .served(&problem.work_item())
            .elements
            .query;
        // Read at another element than the float it is stored in, an operand is bound as its u32
        // words, each one value of that float, which its kernel's stages decode at the read
        // element.
        let words = match stored {
            ElemType::Float(kind) if read != stored => Some(kind),
            _ => None,
        };
        let arg = |tensor: &PlainTensor, axes: &'static [Axis], width: usize, element: ElemType| {
            launcher
                .arg(tensor.binding.clone(), element)
                .axes(axes)
                .vectorize(width)
        };
        let read_operand = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| match words {
            Some(kind) => arg(tensor, axes, width, ElemType::UInt(UIntKind::U32)).packed(kind),
            None => arg(tensor, axes, width, read),
        };
        let query = read_operand(&self.query, &[B, KV, G, Q, D], widths.query)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let cache = match &self.cache {
            AttentionCache::Float { key, value, pages } => {
                // A pool of pages is read as the sequences its table names. The table covers the
                // launch: `paged` states the batch as its rows and the keys as its columns'
                // positions.
                let read_cache = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
                    let arg = read_operand(tensor, axes, width);
                    match pages {
                        Some(_) => arg.paged_covering(B, S),
                        None => arg,
                    }
                    .build()
                    .expect(ADMITTED_PLAN_RUNS)
                };
                let key = match blueprint.microkernel.key_order() {
                    MatmulOrder::InBufferOrder => read_cache(key, &[B, KV, S, D], widths.key),
                    MatmulOrder::Stated => {
                        read_cache(&key.transposed(), &[B, KV, D, S], widths.key)
                    }
                };
                let value = read_cache(value, &[B, KV, S, DV], widths.value);
                AttentionCacheBinding::Tiles { key, value }
            }
            AttentionCache::Packed(cache) => AttentionCacheBinding::Packed(cache.arg()),
        };
        let out = arg(
            &self.out_view,
            &[B, KV, G, Q, DV],
            widths.out,
            problem.elements.out,
        )
        .build()
        .expect(ADMITTED_PLAN_RUNS);
        let pages = match &self.cache {
            AttentionCache::Float {
                pages: Some(pages), ..
            } => ComptimeOptionArgs::Some(pages.binding.clone().into_tensor_arg()),
            _ => ComptimeOptionArgs::None,
        };
        AttentionBinding {
            launcher,
            query,
            cache,
            out,
            lengths: self.lengths.binding.clone().into_tensor_arg(),
            pages,
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

impl AttentionCache {
    /// Key heads.
    fn num_kv_heads(&self) -> usize {
        match self {
            Self::Float { key, .. } => key.shape()[1],
            Self::Packed(cache) => cache.key.words.shape()[1],
        }
    }

    /// Keys each sequence holds, attended or not: a pool's are its table's pages.
    fn seq_kv(&self) -> usize {
        match self {
            Self::Float {
                key,
                pages: Some(pages),
                ..
            } => pages.shape()[1] * key.shape()[2],
            Self::Float { key, .. } => key.shape()[2],
            Self::Packed(cache) => cache.key.words.shape()[2],
        }
    }

    /// Elements of a value.
    fn val_dim(&self) -> usize {
        match self {
            Self::Float { value, .. } => value.shape()[3],
            Self::Packed(cache) => cache.coords(),
        }
    }

    /// How the keys and values lie.
    fn layout(&self) -> AttentionCacheLayout {
        match self {
            Self::Float { pages: None, .. } => AttentionCacheLayout::Sequences,
            Self::Float { key, .. } => AttentionCacheLayout::Pages {
                page_size: key.shape()[2],
            },
            Self::Packed(cache) => AttentionCacheLayout::Packed(cache.layout()),
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
