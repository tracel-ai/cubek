//! Choosing an attention's routine by measurement, in one of two autotunes: a call of many query
//! positions, and a call of one.
//!
//! The routines serve GQA natively, each key head read once per box. They attend the first
//! `len` positions of a cache with the queries bottom-right aligned, so a chunk against a longer
//! cache is raced on the one chunk every chunk key stands for ([`AttentionKey`]).

use crate::kernels::select::selector::QueryPositions;
use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{
    AttentionBlueprint, AttentionRoutine, AttentionRoutineCandidate, AttentionRoutineProblem,
    AttentionScoring, AttentionTensors, CubeShuffleAttentionRoutine, CubeStagedAttentionRoutine,
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchFragmentAttentionRoutine,
    HypercubeSecondLaunchShuffleAttentionRoutine, UnitAttentionRoutine,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::{MemoryAccess, compute_throughput_key};
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, Work, local_tuner};
use cubecl::zspace::{Shape, metadata::Metadata};
use cubecl::{CubeTuneId, Device};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
};
use cubek_std::Hardware;
use cubek_std::Instructions;
use cubek_std::autotune::{Autotune, Ceiling, GroupPriority, Roofline, TuneCandidate};
use cubek_std::routine::Routine;

/// The operands of one float attention, as autotune measures them, and the tensors every
/// candidate launches on, the output among them: a candidate writes the output these inputs hold
/// rather than handing one back, so the call's output is the one the winner wrote.
#[derive(Clone)]
pub struct AttentionTuneInputs {
    client: Client,
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
    tensors: AttentionTensors,
}

/// The attention of `query` over `key` and `value`, `[b, q, h, dv]`, through whichever routine
/// and setting measures fastest for this problem on `device`: among the routines that serve many
/// query positions, or those that serve one, as `positions` says. `routines` is the priority of
/// the routines' group; under it sits the unit routine's lead, which serves any problem on any
/// device.
pub fn launch_tuned(
    positions: QueryPositions,
    device: &Device,
    inputs: AttentionTuneInputs,
    routines: GroupPriority,
) -> TensorHandle {
    let client = inputs.client.clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::new(client.properties_shared(), client.target_properties());
    match positions {
        QueryPositions::Many => {
            // Named rather than `module_path!()`, so the cached winners stay this autotune's
            // wherever the code lives; a cached winner is an index into the candidate list.
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("prefill");
            let tunables = TUNER.init(&id, move || many_queries(&hw, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
        QueryPositions::One => {
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("decode");
            let tunables = TUNER.init(&id, move || one_query(&hw, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
    }
    inputs.tensors.into_out()
}

type AttentionAutotune = Autotune<AttentionKey, AttentionTuneInputs, ()>;

/// The routines that serve many query positions: the cube routines, a cube of plane unit groups
/// only where some float element has no square instruction for the plane fragments (elsewhere the
/// fragments serve every element, faster), and the hypercube routines for a prompt's or a chunk's
/// few boxes over many keys. The unit routine is listed only where none of them can run, for want
/// of a plane width the device commits to: elsewhere measuring its candidates takes longer than
/// the rest of the autotune. The plane routine is not listed.
fn many_queries(
    hw: &Hardware,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::measured_on(
        AttentionTuneInputs::many_queries_key,
        "as-keyed",
        AttentionTuneInputs::many_queries_representative,
    );
    let routines = autotune.group("microkernel", priority);
    autotune = autotune
        .read_cold(AttentionTuneInputs::many_queries_roofline)
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
                hw,
            ),
            launch,
        );
    if !hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .serves_attention_in_every_float()
    {
        autotune = autotune.prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new(),
                hw,
            ),
            launch,
        );
    }
    autotune = autotune
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(CubeShuffleAttentionRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw),
            launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                HypercubeSecondLaunchShuffleAttentionRoutine::new(),
                hw,
            ),
            launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeSecondLaunchFragmentAttentionRoutine, hw),
            launch,
        );
    if hw.plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionRoutineCandidate::list(UnitAttentionRoutine, hw),
            launch,
        );
    }
    autotune.floor(unit_lead(hw)).finish()
}

/// The routines that serve a single position, reading its group of heads as rows. The
/// hypercube's cubes of plane shuffles share each key head's cache, merged on the last arrival or
/// by a second launch; the cube's plane shuffles split a cube's keys. No routine whose cubes stage
/// is listed: a cube staging a step for a decode step's few rows measured twelve times the
/// hypercube's time. As for many positions, the unit routine is listed only where neither can
/// run, and the plane routine not at all.
fn one_query(
    hw: &Hardware,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::measured_on(
        AttentionTuneInputs::one_query_key,
        "at-capacity",
        AttentionTuneInputs::at_capacity,
    );
    let routines = autotune.group("microkernel", priority);
    autotune = autotune
        .read_cold(AttentionTuneInputs::one_query_roofline)
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw),
            launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(
                HypercubeSecondLaunchShuffleAttentionRoutine::new(),
                hw,
            ),
            launch,
        )
        .prioritized(
            &routines,
            AttentionRoutineCandidate::list(CubeShuffleAttentionRoutine, hw),
            launch,
        );
    if hw.plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionRoutineCandidate::list(UnitAttentionRoutine, hw),
            launch,
        );
    }
    autotune.floor(unit_lead(hw)).finish()
}

/// Both autotunes' floor: the unit routine's lead, which asks nothing of the hardware. Nobody
/// wants it to win, but a plan cubecl cannot fill is a panic, and a configuration may take every
/// group out of the autotune.
fn unit_lead(hw: &Hardware) -> Tunable<AttentionKey, AttentionTuneInputs, ()> {
    let hw = hw.clone();
    Tunable::new("unit-attention-lead", move |inputs: AttentionTuneInputs| {
        UnitAttentionRoutine
            .launch_lead_on(&hw, inputs.tensors)
            .map(|_| ())
            .map_err(|why| format!("the unit routine's lead does not serve this problem: {why:?}"))
    })
}

/// `inputs` attended by `candidate`'s routine at its setting into their output, or why it did
/// not run.
fn launch<R, M: AttentionMicrokernel, SM: PartialEq>(
    candidate: &AttentionRoutineCandidate<R>,
    inputs: AttentionTuneInputs,
) -> Result<(), String>
where
    R: AttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy,
{
    let settings = candidate.strategy();
    let hw = Hardware::new(
        inputs.client.properties_shared(),
        inputs.client.target_properties(),
    );
    candidate
        .routine()
        .launch_on(&hw, inputs.tensors, settings)
        .map(|_| ())
        .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
}

impl AttentionTuneInputs {
    /// Attention of `query` (`[b, h, q, d]`) over each sequence's first `lengths[b]` of `key`
    /// (`[b, kv, s, d]`) and `value` (`[b, kv, s, dv]`), or of the keys and values `pages` lays
    /// out in two pools of pages ([`AttentionTensors::paged`]), the output allocated. `arrivals`
    /// are the counters the cubes of a box count themselves in at, held across launches.
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
            tensors,
        })
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

    /// The bytes of the keys and values every sequence attends.
    fn kv_bytes(&self, batch: usize) -> usize {
        let dim = self.key.shape()[3];
        2 * batch * self.kv_heads() * self.keys() * dim * self.key.elem.size()
    }

    /// The key many query positions are tuned under. A chunk — queries the tail of a longer
    /// prefix — is stated as the one chunk every chunk key stands for.
    fn many_queries_key(&self) -> AttentionKey {
        let q = self.query.shape();
        let (queries, keys) = match q[2] == self.keys() {
            true => (q[2], self.keys()),
            false => (AttentionKey::CHUNK_QUERIES, AttentionKey::CHUNK_PREFIX),
        };
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
        )
    }

    /// The key a decode step is tuned under: one query against the cache's capacity. The single
    /// position sees the whole attended prefix, or its window's end of it, so a causal mask is no
    /// part of the key.
    fn one_query_key(&self) -> AttentionKey {
        let q = self.query.shape();
        AttentionKey::new(
            self.query.elem,
            q[1],
            self.kv_heads(),
            q[3],
            1,
            self.keys(),
            q[0],
            self.scoring.visibility,
            self.scoring.scale,
            self.page_size(),
        )
    }

    /// The inputs many query positions are measured on: the problem `key` describes.
    ///
    /// A whole prompt's key stands for every prompt its bucket holds, so the round measures the
    /// bucket's longest ([`whole_representative`](Self::whole_representative)). A chunk's key
    /// stands for every chunk, so the round measures the chunk the key names on operands of its
    /// own rather than whichever chunk reached the autotune first. Uploaded rather than filled:
    /// a process mode that drops launches drops a launch, not an upload.
    fn many_queries_representative(key: &AttentionKey, inputs: &Self) -> Self {
        let Some(problem) = chunk_problem(key) else {
            return inputs.whole_representative(key);
        };
        let client = &inputs.client;
        let elem = inputs.query.elem;
        let head_dim = problem.head_dim;
        let zeros = |shape: [usize; 4]| zeros(client, &shape, contiguous(&shape), elem);
        let query = zeros([problem.batch, problem.heads(), problem.seq_q, head_dim]);
        // A paged key's chunk is read through a table of its page size, each sequence's pages in
        // order, as a pool of its own.
        let (key, value, pages) = match key.page_size() {
            None => {
                let shape = [
                    problem.batch,
                    problem.num_kv_heads,
                    problem.seq_kv,
                    head_dim,
                ];
                (zeros(shape), zeros(shape), None)
            }
            Some(page_size) => {
                let width = problem.seq_kv.div_ceil(page_size);
                let count = problem.batch * width;
                let shape = [count, problem.num_kv_heads, page_size, head_dim];
                let table: Vec<u32> = (0..count as u32).collect();
                (
                    zeros(shape),
                    zeros(shape),
                    Some(upload_u32(client, &[problem.batch, width], &table)),
                )
            }
        };
        inputs.measured_on(query, key, value, pages, None)
    }

    /// A whole prompt's round: zero queries, keys and values as long as the key's bucket, laid
    /// out as the call's are and attended to their end; or the call's own where the cache is
    /// paged or an operand cannot be laid out alike.
    fn whole_representative(&self, key: &AttentionKey) -> Self {
        // A prompt's positions lie along each operand's third axis: `[batch, heads, positions, dim]`.
        const POSITIONS: usize = 2;
        let grown = |tensor: &PlainTensor, extent: usize| {
            zeros_grown(&self.client, tensor, POSITIONS, extent)
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

    /// The inputs a decode candidate is measured on: the call's own operands, attended **to their
    /// capacity**.
    ///
    /// The length is the one input the key does not name — it lives on device so the launch
    /// replays across steps — and a live call's is no fair sample of it: where launches are
    /// discarded it is never advanced, so every candidate would be measured over an empty cache.
    /// The capacity is the depth the key is anchored on and the one
    /// [`one_query_work`](Self::one_query_work) already prices, so the round's bound and its
    /// measurement agree about the work.
    fn at_capacity(_key: &AttentionKey, inputs: &Self) -> Self {
        inputs.measured_on(
            inputs.query.clone(),
            inputs.key.clone(),
            inputs.value.clone(),
            inputs.pages.clone(),
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
        let lengths = upload_u32(&self.client, &[batch], &vec![keys as u32; batch]);
        Self::new(
            &self.client,
            query,
            key,
            value,
            pages,
            lengths,
            arrivals,
            self.scoring,
        )
        .expect("operands of the call's elements and heads are admitted as the call's were")
    }

    fn many_queries_work(&self) -> Work {
        let q = self.query.shape();
        let (batch, heads, q_len, head_dim) = (q[0], q[1], q[2], q[3]);
        Work {
            compute_ops: attention_ops(
                batch,
                heads,
                q_len,
                self.keys(),
                head_dim,
                self.scoring.visibility.is_causal(),
            ),
            // The output is the query's element count laid out heads-adjacent.
            bytes: self.query.binding.handle.size() as usize * 2 + self.kv_bytes(batch),
        }
    }

    fn one_query_work(&self) -> Work {
        let q = self.query.shape();
        let (batch, heads, head_dim) = (q[0], q[1], q[3]);
        Work {
            // One query position, and no mask: the single query sits at the end of the prefix,
            // so every position it scores is one it attends.
            compute_ops: attention_ops(batch, heads, 1, self.keys(), head_dim, false),
            bytes: self.query.binding.handle.size() as usize * 2 + self.kv_bytes(batch),
        }
    }

    /// The many-positions roofline: the instruction. A whole prompt reuses each key and value
    /// across every query position in the block, so what binds is the arithmetic rather than
    /// the bus, at the instruction the planes contract on, and plain arithmetic where they can
    /// contract on none. The instruction is
    /// [`for_attention`](cubek_std::Instructions::for_attention)'s answer, not the offer's: the
    /// planes need a shape for the mix as well as the score.
    fn many_queries_roofline(key: &AttentionKey, inputs: &Self) -> Roofline {
        let elem = key.elem();
        let accumulate = ElemType::Float(FloatKind::F32);
        let hw = Hardware::new(
            inputs.client.properties_shared(),
            inputs.client.target_properties(),
        );
        let instruction = hw
            .properties
            .features
            .matmul
            .cmma
            .iter()
            .copied()
            .for_attention(elem)
            .largest()
            .map(|i| (i.m, i.n, i.k));
        // A chunk key's candidates run on the chunk the key stands for, whatever chunk the call
        // brought, so a chunk key prices its own problem, at the call's element.
        let work = match chunk_problem(key) {
            Some(problem) => problem_work(&problem, inputs.query.elem.size()),
            None => inputs.many_queries_work(),
        };
        Roofline {
            client: inputs.client.clone(),
            ceiling: Ceiling::Compute(compute_throughput_key(instruction, elem, accumulate)),
            work,
        }
    }

    /// The one-position roofline: the bus, in the direction the traffic goes. One query against
    /// the whole cache reuses nothing — every key and value byte is read once and the write is a
    /// single row — so the ceiling is the **read** bandwidth.
    fn one_query_roofline(_key: &AttentionKey, inputs: &Self) -> Roofline {
        Roofline {
            client: inputs.client.clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: inputs.one_query_work(),
        }
    }
}

/// The operations an attention costs: **two** contractions over the same score tile — `q · kᵀ`
/// and `p · v`, each one multiply-accumulate per `(query, position, dim)` and two operations per
/// accumulate — halved where the causal mask means half the tile is never scored. The minimum the
/// problem needs, which is what a roofline is measured against.
fn attention_ops(
    batch: usize,
    heads: usize,
    q_len: usize,
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> usize {
    let ops = 4 * batch * heads * q_len * seq * head_dim;
    match causal {
        true => ops / 2,
        false => ops,
    }
}

/// The problem a chunk key stands for, at its buckets' extents; `None` for a whole prompt, and
/// for a key whose head counts do not divide, which no call produces.
fn chunk_problem(key: &AttentionKey) -> Option<AttentionRoutineProblem> {
    AttentionRoutineProblem::try_from(key)
        .ok()
        .filter(|_| key.chunk())
}

/// The work of a problem rather than of operands: the query and the output at the query heads,
/// the keys and values at the key heads.
fn problem_work(problem: &AttentionRoutineProblem, elem_size: usize) -> Work {
    let heads = problem.heads();
    let causal = problem.visibility.is_causal();
    Work {
        compute_ops: attention_ops(
            problem.batch,
            heads,
            problem.seq_q,
            problem.seq_kv,
            problem.head_dim,
            causal,
        ),
        bytes: problem.batch
            * problem.head_dim
            * elem_size
            * (2 * heads * problem.seq_q + 2 * problem.num_kv_heads * problem.seq_kv),
    }
}

/// Row-major strides over `shape`.
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// A zero-filled `shape` of `elem` at `strides`, uploaded.
fn zeros(client: &Client, shape: &[usize], strides: Vec<usize>, elem: ElemType) -> PlainTensor {
    let elements = shape
        .iter()
        .zip(&strides)
        .map(|(extent, stride)| (extent - 1) * stride)
        .sum::<usize>()
        + 1;
    let handle = client.create_from_slice(&vec![0u8; elements * elem.size()]);
    let meta = Metadata::new(Shape::from(shape.to_vec()), strides);
    PlainTensor::of_handle(TensorHandle::from_metadata(handle, meta, elem))
}

/// `values` as a `u32` tensor of `shape`, uploaded.
fn upload_u32(client: &Client, shape: &[usize], values: &[u32]) -> PlainTensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let handle = client.create_from_slice(&bytes);
    let meta = Metadata::new(Shape::from(shape.to_vec()), contiguous(shape));
    PlainTensor::of_handle(TensorHandle::from_metadata(
        handle,
        meta,
        ElemType::UInt(UIntKind::U32),
    ))
}

/// A zero-filled tensor of `tensor`'s element, its `axis` grown to `extent`, laid out in
/// `tensor`'s order of strides: what a round measures in place of a call shorter than its key's
/// bucket, so the winner is elected on the bucket's longest launch whatever order the calls come
/// in.
///
/// `None` where `extent` does not grow `axis`, or where a fresh buffer cannot repeat the layout:
/// a tiled, strided or broadcast buffer.
fn zeros_grown(
    client: &Client,
    tensor: &PlainTensor,
    axis: usize,
    extent: usize,
) -> Option<PlainTensor> {
    let shape = tensor.shape();
    let shape = shape.as_slice();
    if extent <= shape[axis] || tensor.is_tiled() {
        return None;
    }
    let strides = &tensor.binding.strides[..];
    // The axes innermost first; a dense buffer's strides are the running product of the extents
    // inside them. An axis of one has no step to repeat.
    let mut order: Vec<usize> = (0..shape.len()).collect();
    order.sort_by_key(|&a| (strides[a], shape[a] != 1));
    let mut step = 1;
    for &a in &order {
        if shape[a] != 1 && strides[a] != step {
            return None;
        }
        step *= shape[a];
    }
    let mut grown = shape.to_vec();
    grown[axis] = extent;
    let mut grown_strides = vec![0; grown.len()];
    let mut step = 1;
    for &a in &order {
        grown_strides[a] = step;
        step *= grown[a];
    }
    Some(zeros(client, &grown, grown_strides, tensor.elem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{AttentionVisibility, ScoreScale};

    /// Qwen3-8B prefilling 512 tokens: 32 heads over a 128-wide head dim.
    const SHAPE: (usize, usize, usize, usize, usize) = (1, 32, 512, 512, 128);

    /// **A chunk key's round is priced on the chunk it times** — the one every chunk key stands
    /// for — and not on whichever chunk the call brought.
    #[test]
    fn a_chunk_key_prices_the_chunk_it_times() {
        let key = AttentionKey::new(
            ElemType::Float(FloatKind::F16),
            32,
            8,
            128,
            AttentionKey::CHUNK_QUERIES,
            AttentionKey::CHUNK_PREFIX,
            1,
            AttentionVisibility::Causal,
            ScoreScale::standard(128),
            None,
        );
        let problem = chunk_problem(&key).expect("a chunk key has a problem");
        assert_eq!(problem.seq_kv, key.seq_kv());
        assert!(problem.seq_kv >= AttentionKey::CHUNK_PREFIX);
        let work = problem_work(&problem, 2);
        assert_eq!(
            work.compute_ops,
            attention_ops(1, 32, problem.seq_q, problem.seq_kv, 128, true)
        );
        assert_eq!(
            work.bytes,
            128 * 2 * (2 * 32 * problem.seq_q + 2 * 8 * problem.seq_kv)
        );
    }

    /// **Both contractions, and nothing between them.** A roofline that counted one would be half
    /// the ceiling every candidate is held to, and the round would stop on a candidate at half
    /// the device.
    #[test]
    fn attention_costs_two_contractions() {
        let (batch, heads, q_len, seq, head_dim) = SHAPE;
        let one_contraction = 2 * batch * heads * q_len * seq * head_dim;
        assert_eq!(
            attention_ops(batch, heads, q_len, seq, head_dim, false),
            2 * one_contraction,
        );
    }

    /// A causal prompt scores half the tile, so it costs half the arithmetic.
    #[test]
    fn a_causal_prompt_costs_half() {
        let (batch, heads, q_len, seq, head_dim) = SHAPE;
        assert_eq!(
            attention_ops(batch, heads, q_len, seq, head_dim, true),
            attention_ops(batch, heads, q_len, seq, head_dim, false) / 2,
        );
    }

    /// A decode is one query against the whole cache: the same arithmetic at `q_len` 1, so its
    /// ceiling scales with the capacity and nothing else.
    #[test]
    fn a_decode_costs_one_query_against_the_cache() {
        let (batch, heads, _, capacity, head_dim) = SHAPE;
        let at = |capacity| attention_ops(batch, heads, 1, capacity, head_dim, false);
        assert_eq!(at(2 * capacity), 2 * at(capacity));
    }
}
