//! The operands of one float attention as autotune measures them: the key a call is tuned under,
//! and the inputs a round measures on in its place.

use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{AttentionRoutineProblem, AttentionScoring, AttentionTensors};
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::MemoryAccess;
use cubecl::tile::launch::Refusal;
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
    /// `[b, h, q, d]`.
    query: PlainTensor,
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
    attended: AttendedKeys,
    pub(super) tensors: AttentionTensors,
}

/// How many of the positions its key and value buffers hold a call attends, as the host knows it:
/// what tells a cache from the keys of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttendedKeys {
    /// As many as the lengths say, which live on device: the buffers are a cache, and their
    /// extent is its capacity, which no key reads.
    Lengths,
    /// Every position the buffers hold: they are the call's own keys, and their extent is the
    /// problem's.
    Every,
}

impl AttentionTuneInputs {
    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of `key`
    /// (`[b, kv, s, d]`) and `value` (`[b, kv, s, dv]`), or of the keys and values `pages` lays
    /// out in two pools of pages ([`AttentionTensors::paged`]), the output allocated. `arrivals`
    /// are the counters the cubes of a box count themselves in at, held across launches.
    ///
    /// The buffers are taken for a cache: the lengths live on device, so the key reads neither
    /// them nor the buffers' extent ([`AttentionKey`]). Buffers that are the call's own keys say
    /// so ([`attending_every_position`](Self::attending_every_position)).
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
        arrivals: Option<PlainTensor>,
        scoring: AttentionScoring,
    ) -> Result<Self, Refusal> {
        let tensors = match &pages {
            Some(pages) => AttentionTensors::paged(
                client,
                query.clone(),
                key.clone(),
                value.clone(),
                pages.clone(),
                lengths,
                scoring,
            ),
            None => AttentionTensors::new(
                client,
                query.clone(),
                key.clone(),
                value.clone(),
                lengths,
                scoring,
            ),
        }?
        .with_arrivals(arrivals.clone());
        Ok(Self {
            client: client.clone(),
            query,
            key,
            value,
            pages,
            arrivals,
            scoring,
            mask: None,
            attended: AttendedKeys::Lengths,
            tensors,
        })
    }

    /// The same attention, its buffers the call's own keys, every position of them attended: the
    /// lengths are the buffers' extent, so the key holds that extent as the problem's rather than
    /// standing for a cache's.
    pub fn attending_every_position(mut self) -> Self {
        self.attended = AttendedKeys::Every;
        self
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

    /// The key positions the buffers hold: a sequence's extent, or its pages' positions.
    fn keys(&self) -> usize {
        let positions = self.key.shape()[2];
        match &self.pages {
            Some(pages) => pages.shape()[1] * positions,
            None => positions,
        }
    }

    fn kv_heads(&self) -> usize {
        self.key.shape()[1]
    }

    fn page_size(&self) -> Option<usize> {
        self.pages.as_ref().map(|_| self.key.shape()[2])
    }

    /// The key many query positions are tuned under. Over a cache it is the one chunk every such
    /// key stands for, whatever the cache holds and however many queries the call brought: a
    /// winner measured on that chunk serves every call over a cache. Over keys of the call's own
    /// it keeps the call's extents.
    pub(super) fn many_queries_key(&self) -> AttentionKey {
        let (queries, keys) = match self.attended {
            AttendedKeys::Lengths => (AttentionKey::CHUNK_QUERIES, AttentionKey::CACHE_PREFIX),
            AttendedKeys::Every => (self.query.shape()[2], self.keys()),
        };
        self.key_of(queries, keys)
    }

    /// The key a decode step is tuned under: one query over the keys a decode round attends
    /// ([`one_query_keys`](Self::one_query_keys)). The single position sees the whole attended
    /// prefix, or its window's end of it, so a causal mask is no part of the key.
    pub(super) fn one_query_key(&self) -> AttentionKey {
        self.key_of(1, self.one_query_keys())
    }

    /// The keys a decode step is keyed and measured over: the prefix every key over a cache
    /// stands for, or the call's own keys where every position of them is attended.
    fn one_query_keys(&self) -> usize {
        match self.attended {
            AttendedKeys::Lengths => AttentionKey::CACHE_PREFIX,
            AttendedKeys::Every => self.keys(),
        }
    }

    /// The key of this call's heads, element and scoring, `queries` positions over `keys`.
    fn key_of(&self, queries: usize, keys: usize) -> AttentionKey {
        let q = self.query.shape();
        AttentionKey::new(
            self.query.elem,
            q[1],
            self.kv_heads(),
            q[3],
            queries,
            keys,
            q[0],
            self.scoring.visibility,
            self.scoring.scale,
            self.page_size(),
            self.mask.is_some(),
        )
    }

    /// The inputs many query positions are measured on: the problem `key` describes.
    ///
    /// A whole prompt's key stands for every prompt its bucket holds, so the round measures the
    /// bucket's longest ([`whole_representative`](Self::whole_representative)). A chunk's key
    /// stands for every chunk, so the round measures the chunk the key names on operands of its
    /// own rather than whichever chunk reached the autotune first. Uploaded rather than filled:
    /// a process mode that drops launches drops a launch, not an upload.
    pub(super) fn many_queries_representative(key: &AttentionKey, inputs: &Self) -> Self {
        // A whole prompt's key, or one whose head counts do not divide, which no call produces,
        // states no chunk.
        let chunk = AttentionRoutineProblem::try_from(key)
            .ok()
            .filter(|_| key.chunk());
        let Some(problem) = chunk else {
            return inputs.whole_representative(key);
        };
        let (client, elem) = (&inputs.client, inputs.query.elem);
        let shape = [
            problem.batch,
            problem.heads(),
            problem.seq_q,
            problem.head_dim,
        ];
        let query = PlainTensor::zeros(client, &shape, row_major_contiguous_strides(shape), elem);
        let (key, value, pages) = inputs.zero_keys(key, problem.batch);
        inputs.measured_on(query, key, value, pages, None)
    }

    /// Zero keys and values of `batch` sequences, as many positions as `key` names, of this
    /// call's element and key heads, laid out as `key` says: one buffer per operand, or pools of
    /// pages of the key's page size read through a table of their own, each sequence's pages in
    /// order.
    fn zero_keys(
        &self,
        key: &AttentionKey,
        batch: usize,
    ) -> (PlainTensor, PlainTensor, Option<PlainTensor>) {
        let (client, elem) = (&self.client, self.key.elem);
        let (kv_heads, positions, head_dim) = (self.kv_heads(), key.seq_kv(), key.head_dim());
        let zeros = |shape: [usize; 4]| {
            PlainTensor::zeros(client, &shape, row_major_contiguous_strides(shape), elem)
        };
        match key.page_size() {
            None => {
                let shape = [batch, kv_heads, positions, head_dim];
                (zeros(shape), zeros(shape), None)
            }
            Some(page_size) => {
                let width = positions.div_ceil(page_size);
                let count = batch * width;
                let shape = [count, kv_heads, page_size, head_dim];
                let table: Vec<u32> = (0..count as u32).collect();
                (
                    zeros(shape),
                    zeros(shape),
                    Some(PlainTensor::of_u32s(client, &[batch, width], &table)),
                )
            }
        }
    }

    /// A whole prompt's round: zero queries, keys and values as long as the key's bucket, laid
    /// out as the call's are and attended to their end; or the call's own where the cache is
    /// paged or an operand cannot be laid out alike.
    fn whole_representative(&self, key: &AttentionKey) -> Self {
        // A prompt's positions lie along each operand's third axis: `[batch, heads, positions, dim]`.
        const POSITIONS: usize = 2;
        let grown = |tensor: &PlainTensor, extent: usize| {
            tensor.zeros_grown(&self.client, POSITIONS, extent)
        };
        let keyed = || {
            Some((
                grown(&self.query, key.seq_q())?,
                grown(&self.key, key.seq_kv())?,
                grown(&self.value, key.seq_kv())?,
            ))
        };
        let (query, key, value) = match self.pages {
            None => keyed(),
            Some(_) => None,
        }
        .unwrap_or_else(|| (self.query.clone(), self.key.clone(), self.value.clone()));
        self.measured_on(query, key, value, self.pages.clone(), None)
    }

    /// The inputs a decode candidate is measured on: the call's queries, attended **to the end**
    /// of the keys the key names ([`one_query_keys`](Self::one_query_keys)).
    ///
    /// The length is the one input the key does not name — it lives on device so the launch
    /// replays across steps — and a live call's is no fair sample of it: where launches are
    /// discarded it is never advanced, so every candidate would be measured over an empty cache.
    /// Over a cache, the round measures zero keys and values as deep as the key's prefix,
    /// whatever the cache holds; over the call's own keys, those keys. Either way the depth is the
    /// one [`one_query_work`](Self::one_query_work) prices, so the round's cold read and its
    /// measurement agree about the work. The call's arrival counters are kept: the round has the
    /// call's sequences and heads, so they cover its boxes, and every candidate counts in at them
    /// as production does.
    pub(super) fn one_query_representative(key: &AttentionKey, inputs: &Self) -> Self {
        let (keys, values, pages) = match inputs.attended {
            AttendedKeys::Lengths => inputs.zero_keys(key, inputs.query.shape()[0]),
            AttendedKeys::Every => (
                inputs.key.clone(),
                inputs.value.clone(),
                inputs.pages.clone(),
            ),
        };
        inputs.measured_on(
            inputs.query.clone(),
            keys,
            values,
            pages,
            inputs.arrivals.clone(),
        )
    }

    /// The inputs a round measures on: these operands, every sequence attending all the
    /// positions its buffers hold, into an output of their own, counting in at `arrivals`. The
    /// lengths are uploaded rather than filled: an upload is not a launch, so a
    /// process mode that discards launches keeps it.
    fn measured_on(
        &self,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        pages: Option<PlainTensor>,
        arrivals: Option<PlainTensor>,
    ) -> Self {
        let batch = query.shape()[0];
        let positions = key.shape()[2];
        let keys = match &pages {
            Some(pages) => pages.shape()[1] * positions,
            None => positions,
        };
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
        let measured = Self::new(
            &self.client,
            query,
            key,
            value,
            pages,
            lengths,
            arrivals,
            self.scoring,
        )
        .expect("operands of the call's elements and heads are admitted as the call's were");
        match mask {
            Some(mask) => measured.with_mask(mask),
            None => measured,
        }
    }

    /// What a decode round's step costs: one query over the keys it is measured over
    /// ([`one_query_keys`](Self::one_query_keys)), each key and value read once.
    fn one_query_work(&self) -> Work {
        let q = self.query.shape();
        let (batch, heads, head_dim) = (q[0], q[1], q[3]);
        let keys = self.one_query_keys();
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

    /// The keys of one attention over caches of several sizes, and over its own keys.
    mod keys {
        use super::*;
        use crate::kernels::{AttentionScoring, AttentionVisibility, QueryPlacement};
        use cubecl::ir::{ElemType, FloatKind};
        use cubek_test_utils::client;

        const F16: ElemType = ElemType::Float(FloatKind::F16);
        /// Query heads, key heads and head dim: a GQA group of four.
        const HEADS: (usize, usize, usize) = (8, 2, 64);

        /// `queries` positions of one sequence, causal past one, over `positions` keys laid in
        /// one buffer, or in pages of `page_size` read through a table of their own.
        fn inputs(
            queries: usize,
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
            let lengths = PlainTensor::of_u32s(&client, &[1], &[queries as u32]);
            let visibility = AttentionVisibility::new(queries > 1, None, QueryPlacement::Last);
            let scoring = AttentionScoring::standard(visibility, head_dim);
            AttentionTuneInputs::new(&client, query, key, value, pages, lengths, None, scoring)
                .expect("an even group of a float element")
        }

        /// **No key reads a cache's capacity.** One prompt and one decode step, over caches from
        /// exactly as long as the prompt to far deeper than the prefix a key stands for, tune
        /// under one key each; over pages, under one key that differs only by the page size, a
        /// layout and not a size.
        #[test]
        fn a_cache_of_any_capacity_is_keyed_alike() {
            for queries in [1, 37, 512, 3000] {
                let key = |positions, page_size| {
                    let inputs = inputs(queries, positions, page_size);
                    match queries {
                        1 => inputs.one_query_key(),
                        _ => inputs.many_queries_key(),
                    }
                };
                // The shortest cache a call fits: as long as the prompt itself.
                let shortest = key(queries.max(64), None);
                let queries_keyed = match queries {
                    1 => 1,
                    _ => AttentionKey::CHUNK_QUERIES,
                };
                assert_eq!(
                    (shortest.seq_q(), shortest.seq_kv()),
                    (queries_keyed, AttentionKey::CACHE_PREFIX),
                    "{queries} queries"
                );
                for positions in [4096, 65536] {
                    assert_eq!(key(positions, None), shortest, "{queries} queries");
                }
                let paged = key(4096, Some(16));
                assert_eq!(paged.page_size(), Some(16));
                assert_eq!(key(65536, Some(16)), paged, "{queries} queries");
            }
        }

        /// Buffers that are the call's own keys hold the problem's extent: a decode over 300 keys
        /// and a prompt over 300 queries' own keys are keyed at their buckets, not at a cache's
        /// prefix.
        #[test]
        fn a_calls_own_keys_are_keyed_at_their_extent() {
            let decode = inputs(1, 300, None)
                .attending_every_position()
                .one_query_key();
            assert_eq!((decode.seq_q(), decode.seq_kv()), (1, 512));
            let prompt = inputs(300, 300, None)
                .attending_every_position()
                .many_queries_key();
            assert_eq!((prompt.seq_q(), prompt.seq_kv()), (512, 512));
            assert!(!prompt.chunk());
        }

        /// **Both contractions, and nothing between them.** A roofline that counted one would be
        /// half the ceiling every candidate is held to, and the round would stop on a candidate
        /// at half the device.
        #[test]
        fn a_decode_costs_two_contractions() {
            let (heads, _, head_dim) = HEADS;
            let one_contraction = 2 * heads * AttentionKey::CACHE_PREFIX * head_dim;
            let work = inputs(1, 256, None).one_query_work();
            assert_eq!(work.compute_ops, 2 * one_contraction);
        }

        /// A decode round over a cache is priced over the prefix it is measured over, not over
        /// the cache: its cold read and its measurement agree about the work.
        #[test]
        fn a_decode_over_a_cache_is_priced_at_the_prefix() {
            let (_, kv_heads, head_dim) = HEADS;
            let shallow = inputs(1, 256, None).one_query_work();
            let deep = inputs(1, 65536, None).one_query_work();
            assert_eq!(shallow.bytes, deep.bytes);
            let kv_bytes = 2 * kv_heads * AttentionKey::CACHE_PREFIX * head_dim * F16.size();
            assert!(shallow.bytes > kv_bytes);
        }
    }
}
