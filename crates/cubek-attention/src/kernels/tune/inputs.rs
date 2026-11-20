//! The operands of one float attention as autotune measures them: the key a call is tuned under,
//! and the inputs a round measures on in its place.

use crate::kernels::tune::key::{AttentionKey, AttentionKeyElements};
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionCache, AttentionQueries, AttentionScoring, AttentionTensors,
    KvSource,
};
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::MemoryAccess;
use cubecl::tile::launch::Decline;
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::autotune::{Ceiling, Roofline};

/// The operands of one float attention, as autotune measures them, and the tensors every
/// candidate launches on, the output among them: a candidate writes the output these inputs hold
/// rather than handing one back, so the call's output is the one the winner wrote.
#[derive(Clone)]
pub struct AttentionTuneInputs {
    pub(super) client: Client,
    /// `[b, h, q, d]`, or every sequence's queries end to end, `[Σ q_b, h, d]`, where
    /// `query_runs` says so.
    query: PlainTensor,
    /// Where every sequence's queries lie end to end: their `u32` cumulative counts `[b]`, and the
    /// most queries any sequence has.
    query_runs: Option<(PlainTensor, usize)>,
    /// `[b, kv, s, d]`, or a pool of pages `[pages, kv, page_size, d]`.
    key: PlainTensor,
    /// `[b, kv, s, dv]`, or a pool of pages `[pages, kv, page_size, dv]`.
    value: PlainTensor,
    /// The `u32` `[b, pages_per_sequence]` table naming each sequence's pages in order.
    pages: Option<PlainTensor>,
    /// The `u32` counters the cubes of a box count themselves in at, held across launches.
    arrivals: Option<PlainTensor>,
    scoring: AttentionScoring,
    /// The `[b, h, q, s]` mask hiding keys from the call's queries
    /// ([`with_mask`](Self::with_mask)).
    mask: Option<PlainTensor>,
    /// How deep the call attends, as its caller buckets it: the keys its key holds.
    depth: usize,
    pub(super) tensors: AttentionTensors,
}

impl AttentionTuneInputs {
    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of `key`
    /// (`[b, kv, s, d]`) and `value` (`[b, kv, s, dv]`), or of the keys and values `pages` lays
    /// out in two pools of pages ([`AttentionTensors::paged`]), the output allocated. `arrivals`
    /// are the counters the cubes of a box count themselves in at, held across launches.
    ///
    /// `depth` is how many keys the call attends, at least as many as its deepest sequence, as its
    /// caller buckets it: the lengths live on device so that every call of one bucket is the same
    /// launch, and the key reads `depth` in their place, never the buffers' extent
    /// ([`AttentionKey`]). Over keys of the call's own, every position attended, it is their
    /// extent.
    ///
    /// # Errors
    ///
    /// What [`AttentionTensors::new`] declines whatever the routine.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        pages: Option<PlainTensor>,
        lengths: PlainTensor,
        depth: usize,
        arrivals: Option<PlainTensor>,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
        let kv = KvSource::new(AttentionCache::Float { key, value });
        let kv = match pages {
            Some(pages) => kv.paged(pages),
            None => kv,
        };
        let queries = AttentionQueries::Dense { query, lengths };
        Self::attention(client, queries, kv, depth, arrivals, scoring)
    }

    /// Attention of `queries` over the float keys and values `kv` states, the output allocated
    /// ([`AttentionTensors::attention`]); otherwise as [`new`](Self::new). A source listing blocks
    /// of keys is keyed and measured as the pages of one block each that the routines read
    /// ([`KvSource::read_as_block_pages`]).
    ///
    /// # Errors
    ///
    /// What [`AttentionTensors::attention`] declines whatever the routine, and
    /// [`ATTENTION_PACKED_UNREAD`]: autotune measures attentions over float keys and values.
    pub fn attention(
        client: &Client,
        queries: AttentionQueries,
        kv: KvSource,
        depth: usize,
        arrivals: Option<PlainTensor>,
        scoring: AttentionScoring,
    ) -> Result<Self, Decline> {
        let kv = kv.read_as_block_pages(client, scoring.visibility)?;
        let AttentionCache::Float { key, value } = kv.cache.clone() else {
            return Err(ATTENTION_PACKED_UNREAD);
        };
        let (query, query_runs) = match &queries {
            AttentionQueries::Dense { query, .. } => (query.clone(), None),
            AttentionQueries::Packed {
                query,
                ends,
                longest,
                ..
            } => (query.clone(), Some((ends.clone(), *longest))),
        };
        let pages = kv.pages.clone();
        let tensors = AttentionTensors::attention(client, queries, kv, scoring)?
            .with_arrivals(arrivals.clone());
        Ok(Self {
            client: client.clone(),
            query,
            query_runs,
            key,
            value,
            pages,
            arrivals,
            scoring,
            mask: None,
            depth,
            tensors,
        })
    }

    /// The tensors every candidate launches on, the output among them.
    pub fn tensors(&self) -> &AttentionTensors {
        &self.tensors
    }

    /// The same attention, each query not seeing the keys `mask` hides
    /// ([`AttentionTensors::with_mask`]).
    pub fn with_mask(mut self, mask: PlainTensor) -> Self {
        self.tensors = self.tensors.with_mask(mask.clone());
        self.mask = Some(mask);
        self
    }

    /// The same attention, writing beside its output each query's logsumexp
    /// ([`AttentionTensors::with_logsumexp`]): every candidate writes it, so the winner's is the
    /// call's.
    pub fn with_logsumexp(mut self) -> Self {
        self.tensors = self.tensors.with_logsumexp();
        self
    }

    /// Each query's logsumexp, `[b, h, q]`, where the attention writes it.
    pub fn logsumexp(&self) -> Option<TensorHandle> {
        self.tensors.logsumexp()
    }

    fn kv_heads(&self) -> usize {
        self.key.shape()[1]
    }

    /// Sequences: the query's batch, or the runs its queries lie in.
    fn sequences(&self) -> usize {
        match &self.query_runs {
            Some((ends, _)) => ends.shape()[0],
            None => self.query.shape()[0],
        }
    }

    /// Query positions of a sequence: the query's, or the longest run.
    fn queries(&self) -> usize {
        match &self.query_runs {
            Some((_, longest)) => *longest,
            None => self.query.shape()[2],
        }
    }

    /// Query heads, and the elements of a query: the query's second and last dims either way.
    fn heads_and_dim(&self) -> (usize, usize) {
        let q = self.query.shape();
        (q[1], q[q.rank() - 1])
    }

    fn page_size(&self) -> Option<usize> {
        self.pages.as_ref().map(|_| self.key.shape()[2])
    }

    /// The key many query positions are tuned under: the call's queries over its depth.
    pub(super) fn many_queries_key(&self) -> AttentionKey {
        self.key_of(self.queries(), self.depth)
    }

    /// The key a decode step is tuned under: one query over the call's depth. The single position
    /// sees the whole attended prefix, or its window's end of it, so a causal mask is no part of
    /// the key.
    pub(super) fn one_query_key(&self) -> AttentionKey {
        self.key_of(1, self.depth)
    }

    /// The key of this call's heads, elements and scoring, `queries` positions over `keys`.
    fn key_of(&self, queries: usize, keys: usize) -> AttentionKey {
        let (heads, head_dim) = self.heads_and_dim();
        AttentionKey::new(
            AttentionKeyElements {
                query: self.query.elem,
                key: self.key.elem,
                value: self.value.elem,
            },
            heads,
            self.kv_heads(),
            head_dim,
            queries,
            keys,
            self.sequences(),
            self.scoring.visibility,
            self.scoring.scale,
            self.page_size(),
            self.mask.is_some(),
            self.query_runs.is_some(),
        )
    }

    /// The inputs many query positions are measured on: the call's sequences, each with the
    /// key's queries over the key's keys, attended to their end. The key stands for every call
    /// its buckets hold, so the round measures the buckets' extents rather than whichever call
    /// reached the autotune first. Uploaded rather than filled: a process mode that drops
    /// launches drops a launch, not an upload.
    pub(super) fn many_queries_representative(key: &AttentionKey, inputs: &Self) -> Self {
        let (query, query_runs) = inputs.queries_at(key.seq_q());
        let (keys, values, pages) = inputs.keys_at(key);
        inputs.measured_on(query, query_runs, keys, values, pages, None)
    }

    /// The inputs a decode candidate is measured on: the call's queries, attended **to the end**
    /// of zero keys and values as deep as the key's keys.
    ///
    /// The lengths are the one input the key does not hold — they live on device so the launch
    /// replays across steps — and a live call's are no fair sample of them: where launches are
    /// discarded they never advance, so every candidate would be measured over an empty cache. The
    /// depth measured is the one [`one_query_work`](Self::one_query_work) prices, so the round's
    /// cold read and its measurement agree about the work. The call's arrival counters are kept:
    /// the round has the call's sequences and heads, so they cover its boxes, and every candidate
    /// counts in at them as production does.
    pub(super) fn one_query_representative(key: &AttentionKey, inputs: &Self) -> Self {
        let (keys, values, pages) = inputs.keys_at(key);
        inputs.measured_on(
            inputs.query.clone(),
            inputs.query_runs.clone(),
            keys,
            values,
            pages,
            inputs.arrivals.clone(),
        )
    }

    /// The queries a round measures, `queries` a sequence: dense ones laid out as the call's
    /// ([`at_positions`](Self::at_positions)), or zero runs of `queries` each, one a sequence,
    /// where the call's lie end to end.
    fn queries_at(&self, queries: usize) -> (PlainTensor, Option<(PlainTensor, usize)>) {
        if self.query_runs.is_none() {
            return (self.at_positions(&self.query, queries), None);
        }
        let (batch, (heads, dim)) = (self.sequences(), self.heads_and_dim());
        let shape = [batch * queries, heads, dim];
        let strides = row_major_contiguous_strides(shape);
        let query = PlainTensor::zeros(&self.client, &shape, strides, self.query.elem);
        let ends: Vec<u32> = (1..=batch).map(|b| (b * queries) as u32).collect();
        let ends = PlainTensor::of_u32s(&self.client, &[batch], &ends);
        (query, Some((ends, queries)))
    }

    /// The keys and values a round measures, as deep as `key`'s keys: laid out as the call's
    /// ([`at_positions`](Self::at_positions)), or zero pools of pages of the key's page size read
    /// through a table of their own, each sequence's pages in order.
    fn keys_at(&self, key: &AttentionKey) -> (PlainTensor, PlainTensor, Option<PlainTensor>) {
        let positions = key.seq_kv();
        let Some(page_size) = key.page_size() else {
            let (keys, values) = (&self.key, &self.value);
            return (
                self.at_positions(keys, positions),
                self.at_positions(values, positions),
                None,
            );
        };
        let client = &self.client;
        let batch = self.sequences();
        let width = positions.div_ceil(page_size);
        let count = batch * width;
        let zeros = |side: &PlainTensor| {
            let shape = [count, self.kv_heads(), page_size, side.shape()[3]];
            PlainTensor::zeros(
                client,
                &shape,
                row_major_contiguous_strides(shape),
                side.elem,
            )
        };
        let table: Vec<u32> = (0..count as u32).collect();
        (
            zeros(&self.key),
            zeros(&self.value),
            Some(PlainTensor::of_u32s(client, &[batch, width], &table)),
        )
    }

    /// `operand`, a `[batch, heads, positions, dim]` operand of the call, at `positions`
    /// positions: itself where it holds as many, zeros laid out as it is where it grows to them,
    /// and contiguous zeros otherwise, as for a cache whose capacity outruns them.
    fn at_positions(&self, operand: &PlainTensor, positions: usize) -> PlainTensor {
        // A sequence's positions lie along each operand's third dimension.
        const POSITIONS: usize = 2;
        if operand.shape()[POSITIONS] == positions {
            return operand.clone();
        }
        operand
            .zeros_grown(&self.client, POSITIONS, positions)
            .unwrap_or_else(|| {
                let mut shape = operand.shape().dims::<4>();
                shape[POSITIONS] = positions;
                let strides = row_major_contiguous_strides(shape);
                PlainTensor::zeros(&self.client, &shape, strides, operand.elem)
            })
    }

    /// The inputs a round measures on: these operands, every sequence attending all the
    /// positions its buffers hold, into an output of their own, counting in at `arrivals`. The
    /// lengths are uploaded rather than filled: an upload is not a launch, so a
    /// process mode that discards launches keeps it.
    fn measured_on(
        &self,
        query: PlainTensor,
        query_runs: Option<(PlainTensor, usize)>,
        key: PlainTensor,
        value: PlainTensor,
        pages: Option<PlainTensor>,
        arrivals: Option<PlainTensor>,
    ) -> Self {
        let batch = match &query_runs {
            Some((ends, _)) => ends.shape()[0],
            None => query.shape()[0],
        };
        let keys = positions_held(&key, pages.as_ref());
        let lengths = PlainTensor::of_u32s(&self.client, &[batch], &vec![keys as u32; batch]);
        // A mask's cells change which scores count, not what a launch reads or how many steps it
        // takes: one cell of the call's element, at stride zero over the measured scores.
        let mask = self.mask.as_ref().map(|mask| {
            let [batch, heads, queries, _] = query.shape().dims::<4>();
            PlainTensor::zeros(
                &self.client,
                &[batch, heads, queries, keys],
                vec![0; 4],
                mask.elem,
            )
        });
        let queries = match query_runs {
            Some((ends, longest)) => AttentionQueries::Packed {
                query,
                ends,
                longest,
                lengths,
            },
            None => AttentionQueries::Dense { query, lengths },
        };
        let kv = KvSource::new(AttentionCache::Float { key, value });
        let kv = match pages {
            Some(pages) => kv.paged(pages),
            None => kv,
        };
        let measured = Self::attention(&self.client, queries, kv, keys, arrivals, self.scoring)
            .expect("operands of the call's elements and heads are admitted as the call's were");
        match mask {
            Some(mask) => measured.with_mask(mask),
            None => measured,
        }
    }

    /// What a decode round's step costs: one query over the keys `key` holds, which the round
    /// measures, each key and value read once.
    fn one_query_work(&self, key: &AttentionKey) -> Work {
        let batch = self.sequences();
        let (heads, head_dim) = self.heads_and_dim();
        let keys = key.seq_kv();
        let kv_bytes =
            2 * batch * self.kv_heads() * keys * self.key.shape()[3] * self.key.elem.size();
        Work {
            // **Two** contractions over the same scores, `q · kᵀ` and `p · v`, each one
            // multiply-accumulate, two operations, per key and dim: the minimum the problem needs,
            // which is what a roofline is measured against. One query position, and no mask: the
            // single query sits at the end of the prefix, so every position it scores is one it
            // attends.
            compute_ops: 4 * batch * heads * keys * head_dim,
            bytes: self.query.binding.handle.size() as usize * 2 + kv_bytes,
        }
    }

    /// The one-position roofline: the bus, in the direction the traffic goes. One query against
    /// the whole cache reuses nothing — every key and value byte is read once and the write is a
    /// single row — so the ceiling is the **read** bandwidth.
    pub(super) fn one_query_roofline(key: &AttentionKey, inputs: &Self) -> Roofline {
        Roofline {
            client: inputs.client.clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: inputs.one_query_work(key),
        }
    }
}

/// The key positions a sequence's buffers hold: `key`'s extent, or its pages' positions.
fn positions_held(key: &PlainTensor, pages: Option<&PlainTensor>) -> usize {
    let positions = key.shape()[2];
    match pages {
        Some(pages) => pages.shape()[1] * positions,
        None => positions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keys of attentions at several depths, over caches of several sizes.
    mod keys {
        use super::*;
        use crate::kernels::{AttentionScoring, AttentionVisibility, QueryPlacement};
        use core::num::NonZeroUsize;
        use cubecl::ir::{ElemType, FloatKind};
        use cubek_test_utils::{TestOutcome, ValidationResult, client};

        const F16: ElemType = ElemType::Float(FloatKind::F16);
        /// Query heads, key heads and head dim: a GQA group of four.
        const HEADS: (usize, usize, usize) = (8, 2, 64);

        /// `queries` positions of one sequence, causal past one, attending `depth` keys of a cache
        /// of `positions` laid in one buffer, or in pages of `page_size` read through a table of
        /// their own.
        fn inputs(
            queries: usize,
            depth: usize,
            positions: usize,
            page_size: Option<usize>,
        ) -> AttentionTuneInputs {
            let client = client();
            let (heads, kv_heads, head_dim) = HEADS;
            let zeros = |shape: [usize; 4]| {
                PlainTensor::zeros(&client, &shape, row_major_contiguous_strides(shape), F16)
            };
            let query = zeros([1, heads, queries, head_dim]);
            let (key, value, pages) = match page_size {
                None => {
                    let shape = [1, kv_heads, positions, head_dim];
                    (zeros(shape), zeros(shape), None)
                }
                Some(page_size) => {
                    let width = positions / page_size;
                    let shape = [width, kv_heads, page_size, head_dim];
                    let table: Vec<u32> = (0..width as u32).collect();
                    let pages = PlainTensor::of_u32s(&client, &[1, width], &table);
                    (zeros(shape), zeros(shape), Some(pages))
                }
            };
            let lengths = PlainTensor::of_u32s(&client, &[1], &[depth as u32]);
            let visibility = AttentionVisibility::new(queries > 1, None, QueryPlacement::Last);
            let scoring = AttentionScoring::standard(visibility, head_dim);
            AttentionTuneInputs::new(
                &client, query, key, value, pages, lengths, depth, None, scoring,
            )
            .expect("an even group of a float element")
        }

        /// The key of `inputs`, one query's or many's.
        fn key(inputs: &AttentionTuneInputs) -> AttentionKey {
            match inputs.queries() {
                1 => inputs.one_query_key(),
                _ => inputs.many_queries_key(),
            }
        }

        /// **A key holds the depth its call attends.** A chunk of 512 queries and a decode step,
        /// each at 512 keys and at 32768, over one cache, are keyed apart, at their depths.
        #[test]
        fn a_call_is_keyed_at_its_depth() {
            for queries in [1, 512] {
                let shallow = key(&inputs(queries, 512, 65536, None));
                let deep = key(&inputs(queries, 32768, 65536, None));
                assert_eq!((shallow.seq_q(), shallow.seq_kv()), (queries, 512));
                assert_eq!((deep.seq_q(), deep.seq_kv()), (queries, 32768));
                assert_ne!(shallow, deep, "{queries} queries");
            }
        }

        /// **Every call of one bucket is the same call to autotune and to its launch.** Depths
        /// from just past one power of two to the next share one key, so one winner, and the
        /// routines prepare one problem for them, so the launch a graph captured replays at each.
        #[test]
        fn a_bucket_is_one_key_and_one_launch() {
            for queries in [1, 37, 512] {
                let calls: Vec<_> = [1025, 1500, 2048]
                    .map(|depth| inputs(queries, depth, 4096, None))
                    .into_iter()
                    .collect();
                let first = &calls[0];
                for call in &calls[1..] {
                    assert_eq!(key(call), key(first), "{queries} queries");
                    assert_eq!(
                        call.tensors().problem(),
                        first.tensors().problem(),
                        "{queries} queries"
                    );
                }
                let next = inputs(queries, 2049, 4096, None);
                assert_ne!(key(&next), key(first), "{queries} queries");
            }
        }

        /// **No key reads a cache's capacity.** One depth over caches from as deep as the call to
        /// far deeper tunes under one key; over pages, under one key that differs only by the
        /// page size, a layout and not a size.
        #[test]
        fn a_cache_of_any_capacity_is_keyed_alike() {
            for queries in [1, 37, 512] {
                let depth = 1024;
                let shortest = key(&inputs(queries, depth, depth, None));
                assert_eq!(key(&inputs(queries, depth, 65536, None)), shortest);
                let paged = key(&inputs(queries, depth, 4096, Some(16)));
                assert_eq!(paged.page_size(), Some(16));
                assert_eq!(key(&inputs(queries, depth, 65536, Some(16))), paged);
            }
        }

        /// A source listing blocks of keys is keyed as the pages of one block each the routines
        /// read, not as the cache it lists them from.
        #[test]
        fn listed_blocks_are_keyed_as_the_pages_read() {
            let client = client();
            let (heads, kv_heads, head_dim) = HEADS;
            let zeros = |shape: [usize; 4]| {
                PlainTensor::zeros(&client, &shape, row_major_contiguous_strides(shape), F16)
            };
            let block = NonZeroUsize::new(4).expect("a block holds a position");
            let kv = KvSource::new(AttentionCache::Float {
                key: zeros([1, kv_heads, 32, head_dim]),
                value: zeros([1, kv_heads, 32, head_dim]),
            })
            .indexed(PlainTensor::of_u32s(&client, &[1, 3], &[5, 0, 6]), block);
            let queries = AttentionQueries::Dense {
                query: zeros([1, heads, 1, head_dim]),
                lengths: PlainTensor::of_u32s(&client, &[1], &[12]),
            };
            let scoring = AttentionScoring::standard(AttentionVisibility::Attended, head_dim);
            let inputs = AttentionTuneInputs::attention(&client, queries, kv, 12, None, scoring)
                .expect("listed blocks within a page, attended whole");
            assert_eq!(inputs.one_query_key().page_size(), Some(4));
        }

        /// **A round measures its key's extents, whatever the call brought.** A chunk of 300
        /// queries at a depth of 1500 over a cache of 4096 is measured as 512 queries over 2048
        /// keys; a decode step at the same depth over 2048 keys; a prompt over its own 300 keys
        /// as 512 over 512, laid out as its own.
        #[test]
        fn a_round_measures_its_keys_extents() {
            let chunk = inputs(300, 1500, 4096, None);
            let round =
                AttentionTuneInputs::many_queries_representative(&chunk.many_queries_key(), &chunk);
            assert_eq!(
                (
                    round.queries(),
                    positions_held(&round.key, round.pages.as_ref())
                ),
                (512, 2048)
            );

            let decode = inputs(1, 1500, 4096, None);
            let round =
                AttentionTuneInputs::one_query_representative(&decode.one_query_key(), &decode);
            assert_eq!(
                (
                    round.queries(),
                    positions_held(&round.key, round.pages.as_ref())
                ),
                (1, 2048)
            );

            let prompt = inputs(300, 300, 300, None);
            let round = AttentionTuneInputs::many_queries_representative(
                &prompt.many_queries_key(),
                &prompt,
            );
            assert_eq!(
                (
                    round.queries(),
                    positions_held(&round.key, round.pages.as_ref())
                ),
                (512, 512)
            );
        }

        /// A paged round measures pools of pages as deep as its key, each sequence's in order.
        #[test]
        fn a_paged_round_measures_pages_as_deep_as_its_key() {
            let decode = inputs(1, 1500, 4096, Some(16));
            let key = decode.one_query_key();
            let round = AttentionTuneInputs::one_query_representative(&key, &decode);
            assert_eq!(
                (
                    round.page_size(),
                    positions_held(&round.key, round.pages.as_ref())
                ),
                (Some(16), 2048)
            );
        }

        /// **Each side of a cache is keyed and measured at its own element.** A chunk and a decode
        /// step over `f16` keys and values, over `e4m3` keys beside `f16` values, and over both
        /// `e4m3`, tune under three keys: a winner over one reads the others' operands differently
        /// or not at all. Each round measures every side at the element its call reads it at. A
        /// cache whose element the runtime declines to store is a skip, not a failure.
        #[test]
        fn each_side_of_a_cache_is_keyed_and_measured_at_its_own_element() {
            let e4m3 = ElemType::Float(FloatKind::E4M3);
            let client = client();
            let (heads, kv_heads, head_dim) = HEADS;
            let over = |queries: usize, key: ElemType, value: ElemType| {
                let zeros = |shape: [usize; 4], elem| {
                    PlainTensor::zeros(&client, &shape, row_major_contiguous_strides(shape), elem)
                };
                let cache = [1, kv_heads, 4096, head_dim];
                let lengths = PlainTensor::of_u32s(&client, &[1], &[1024]);
                let visibility = AttentionVisibility::new(queries > 1, None, QueryPlacement::Last);
                AttentionTuneInputs::new(
                    &client,
                    zeros([1, heads, queries, head_dim], F16),
                    zeros(cache, key),
                    zeros(cache, value),
                    None,
                    lengths,
                    1024,
                    None,
                    AttentionScoring::standard(visibility, head_dim),
                )
            };
            for queries in [1, 512] {
                let caches = [(F16, F16), (e4m3, F16), (e4m3, e4m3)].map(|(key, value)| {
                    let inputs = match over(queries, key, value) {
                        Err(Decline::Unserved { why }) => return Err(why),
                        inputs => inputs.expect("an even group of float elements"),
                    };
                    let (keyed, round) = match queries {
                        1 => {
                            let keyed = inputs.one_query_key();
                            let round =
                                AttentionTuneInputs::one_query_representative(&keyed, &inputs);
                            (keyed, round)
                        }
                        _ => {
                            let keyed = inputs.many_queries_key();
                            let round =
                                AttentionTuneInputs::many_queries_representative(&keyed, &inputs);
                            (keyed, round)
                        }
                    };
                    assert_eq!(
                        (round.query.elem, round.key.elem, round.value.elem),
                        (F16, key, value),
                        "{queries} queries: the round measures another cache"
                    );
                    Ok(keyed)
                });
                let keyed: Vec<_> = caches
                    .into_iter()
                    .filter_map(|cache| {
                        cache
                            .inspect_err(|why| {
                                TestOutcome::Validated(ValidationResult::Skipped(why.to_string()))
                                    .enforce()
                            })
                            .ok()
                    })
                    .collect();
                for (index, key) in keyed.iter().enumerate() {
                    for other in &keyed[index + 1..] {
                        assert_ne!(key, other, "{queries} queries");
                    }
                }
            }
        }

        /// **Both contractions, and nothing between them.** A roofline that counted one would be
        /// half the ceiling every candidate is held to, and the round would stop on a candidate
        /// at half the device.
        #[test]
        fn a_decode_costs_two_contractions() {
            let (heads, _, head_dim) = HEADS;
            let decode = inputs(1, 256, 4096, None);
            let work = decode.one_query_work(&decode.one_query_key());
            let one_contraction = 2 * heads * 256 * head_dim;
            assert_eq!(work.compute_ops, 2 * one_contraction);
        }

        /// A decode round is priced at its key's depth, not at the cache: its cold read and its
        /// measurement agree about the work.
        #[test]
        fn a_decode_is_priced_at_its_keys_depth() {
            let (_, kv_heads, head_dim) = HEADS;
            let small = inputs(1, 2048, 2048, None);
            let large = inputs(1, 2048, 65536, None);
            let price =
                |inputs: &AttentionTuneInputs| inputs.one_query_work(&inputs.one_query_key());
            assert_eq!(price(&small).bytes, price(&large).bytes);
            let kv_bytes = 2 * kv_heads * 2048 * head_dim * F16.size();
            assert!(price(&small).bytes > kv_bytes);
        }
    }
}
