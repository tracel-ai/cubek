//! The operands of one float attention as autotune measures them: the key a call is tuned under,
//! and what a round on the call's own buffers costs.

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
        let tensors =
            AttentionTensors::attention(client, queries, kv, scoring)?.with_arrivals(arrivals);
        Ok(Self {
            client: client.clone(),
            query,
            query_runs,
            key,
            value,
            pages,
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
    ///
    /// # Errors
    ///
    /// As [`AttentionTensors::with_mask`].
    pub fn with_mask(mut self, mask: PlainTensor) -> Result<Self, Decline> {
        self.tensors = self.tensors.with_mask(mask.clone())?;
        self.mask = Some(mask);
        Ok(self)
    }

    /// The same attention, writing beside its output each query's logsumexp
    /// ([`AttentionTensors::with_logsumexp`]): every candidate writes it, so the winner's is the
    /// call's.
    ///
    /// # Errors
    ///
    /// As [`AttentionTensors::with_logsumexp`].
    pub fn with_logsumexp(mut self) -> Result<Self, Decline> {
        self.tensors = self.tensors.with_logsumexp()?;
        Ok(self)
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

    /// The key of this call's heads, elements and scoring, `queries` positions over `keys`, each at
    /// its bucket.
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
            AttentionKey::positions_bucket(queries),
            AttentionKey::positions_bucket(keys),
            self.sequences(),
            self.scoring.visibility,
            self.scoring.scale,
            self.page_size(),
            self.mask.is_some(),
            self.query_runs.is_some(),
        )
    }

    /// The key positions of a sequence the call can read: its depth, or its buffers' positions
    /// where a caller's rounding takes the depth past them.
    fn positions_read(&self) -> usize {
        let positions = self.key.shape()[2];
        let held = match &self.pages {
            Some(pages) => pages.shape()[1] * positions,
            None => positions,
        };
        self.depth.min(held)
    }

    /// What a decode round's step costs: one query over the positions the call can read, which
    /// the round runs on, each key and value read once.
    fn one_query_work(&self) -> Work {
        let batch = self.sequences();
        let (heads, head_dim) = self.heads_and_dim();
        let keys = self.positions_read();
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
    pub(super) fn one_query_roofline(_key: &AttentionKey, inputs: &Self) -> Roofline {
        Roofline {
            client: inputs.client.clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: inputs.one_query_work(),
        }
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
        use cubecl::zspace::striding::row_major_contiguous_strides;
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

        /// **Each side of a cache is keyed at its own element.** A chunk and a decode step over
        /// `f16` keys and values, over `e4m3` keys beside `f16` values, and over both `e4m3`, tune
        /// under three keys: a winner over one reads the others' operands differently or not at
        /// all. A cache whose element the runtime declines to store is a skip, not a failure.
        #[test]
        fn each_side_of_a_cache_is_keyed_at_its_own_element() {
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
                    let keyed = match queries {
                        1 => inputs.one_query_key(),
                        _ => inputs.many_queries_key(),
                    };
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
            let work = inputs(1, 256, 4096, None).one_query_work();
            let one_contraction = 2 * heads * 256 * head_dim;
            assert_eq!(work.compute_ops, 2 * one_contraction);
        }

        /// A decode over a cache is priced at its depth, not at the cache.
        #[test]
        fn a_decode_is_priced_at_its_depth() {
            let (_, kv_heads, head_dim) = HEADS;
            let small = inputs(1, 2048, 2048, None).one_query_work();
            let large = inputs(1, 2048, 65536, None).one_query_work();
            assert_eq!(small.bytes, large.bytes);
            let kv_bytes = 2 * kv_heads * 2048 * head_dim * F16.size();
            assert!(small.bytes > kv_bytes);
        }

        /// **A decode is priced at no more than the round reads.** A depth of 300 is priced at
        /// 300 though its key holds 512, and a depth its caller rounded up to 512 at the positions
        /// its buffers hold, 300 in one buffer or 320 in pages.
        #[test]
        fn a_decode_is_priced_within_its_depth_and_its_buffers() {
            let read = |depth, positions, page_size| {
                inputs(1, depth, positions, page_size).positions_read()
            };
            assert_eq!(read(300, 4096, None), 300);
            assert_eq!(read(512, 300, None), 300);
            assert_eq!(read(512, 320, Some(16)), 320);
        }
    }
}
