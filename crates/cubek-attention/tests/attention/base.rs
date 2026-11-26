//! What every attention routine computes: `out = softmax(scale · q · kᵀ) · v` over the keys each
//! query sees, written `[b, q, h, dv]`, and each query's `ln Σ exp(score)` over them, `[b, h, q]`.
//! The cases are chosen to be awkward: no step of queries or keys divides them, the queries sit
//! among the keys last or first, and a mask differs from head to head.

use core::fmt::Debug;
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, shape};
use cubek_attention::kernels::{
    AttentionCache, AttentionLaunch, AttentionQueries, AttentionScoring, AttentionTensors,
    CacheScales, DirectAttention, KvSource, ScoreScale, SplitKeysAttention,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeChecker, TestInput, Written};
use half::f16;

/// The shape of one attention, and how its queries see its keys.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionCase {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
    pub val_dim: usize,
    pub visibility: AttentionVisibility,
    /// A mask's pattern, where the case has one: query `i` of head `h` in sequence `b` does not
    /// see key `j` where `hidden([b, h, i, j])` holds.
    pub hidden: Option<fn([usize; 4]) -> bool>,
    /// The element the cache stores its keys in: `f16` as the query, or `e4m3`, an FP8 cache's.
    pub keys: FloatKind,
    /// The element the cache stores its values in, as [`keys`](Self::keys) its keys.
    pub values: FloatKind,
    /// The one scale each side of the cache is stored under, where it has them: the attention
    /// reads the stored keys and values times their side's scale.
    pub scales: Option<KvScales>,
    /// The positions a page holds where the cache lies in pools of pages, its sequences' pages in
    /// another order than their positions. `None` where each sequence lies whole.
    pub page_size: Option<NonZeroUsize>,
    /// Each row's sink, where the case has sinks: row `i` of head `h` in sequence `b` also weighs
    /// `sink([b, h, i])` at a value of zero.
    pub sinks: Option<fn([usize; 3]) -> f32>,
    /// A relative bias by distance, where the case has one.
    pub relative: Option<RelativeCase>,
    /// Positions the cache's buffers hold past the attended keys: a partly filled cache, whose
    /// spare positions no attention may read.
    pub spare: usize,
}

/// A relative bias by distance: a score query `i` of head `h` in sequence `b` takes of a key it
/// sees `δ` positions back takes `bias([b, h, i, δ])` where `δ < extent`, nothing otherwise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RelativeCase {
    pub extent: usize,
    pub bias: fn([usize; 4]) -> f32,
}

/// A relative bias of an extent shorter than the attended keys, so the farthest keys take none.
pub(crate) const RELATIVE_SHORT: RelativeCase = RelativeCase {
    extent: 7,
    bias: AttentionCase::relative_pattern,
};

/// A relative bias of an extent past every distance a query reaches.
pub(crate) const RELATIVE_LONG: RelativeCase = RelativeCase {
    extent: 50,
    bias: AttentionCase::relative_pattern,
};

/// The scale over every key, and the one over every value, a cache quantized per tensor is
/// stored under.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvScales {
    pub key: f32,
    pub value: f32,
}

/// Scales of a cache quantized per tensor: neither is 1, and the keys' is not the values'.
pub(crate) const PER_TENSOR: KvScales = KvScales {
    key: 0.37,
    value: 2.5,
};

/// A prompt chunk of two groups of two query heads: no step of queries or keys divides it evenly.
pub(crate) const PROMPT: AttentionCase = AttentionCase {
    batch: 1,
    heads: 4,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 37,
    head_dim: 32,
    val_dim: 32,
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::Last,
    },
    hidden: None,
    keys: FloatKind::F16,
    values: FloatKind::F16,
    scales: None,
    page_size: None,
    sinks: None,
    relative: None,
    spare: 0,
};

/// A causal prompt chunk at a head dim of 128, the one a model's prefill runs: four query heads to
/// a key head, and queries and keys no box of either divides.
pub(crate) const PROMPT_WIDE: AttentionCase = AttentionCase {
    heads: 8,
    kv_heads: 2,
    seq_q: 70,
    seq_kv: 300,
    head_dim: 128,
    val_dim: 128,
    ..PROMPT
};

/// The wide prompt at a head dim of 256, six query heads to a key head: Qwen3.8 27B's heads on
/// one card of four under tensor parallelism, whose output a plane holds sixteen fragments wide.
pub(crate) const PROMPT_HEAD_DIM_256: AttentionCase = AttentionCase {
    heads: 6,
    kv_heads: 1,
    head_dim: 256,
    val_dim: 256,
    ..PROMPT_WIDE
};

/// A decode step at a model's head dim: one position of the wide prompt's heads.
pub(crate) const DECODE_WIDE: AttentionCase = AttentionCase {
    seq_q: 1,
    visibility: AttentionVisibility::Attended,
    ..PROMPT_WIDE
};

/// `case` over each FP8 cache: `e4m3` keys, `e4m3` values, and both, beside the `f16` query.
pub(crate) const fn fp8_caches(case: AttentionCase) -> [AttentionCase; 3] {
    [
        AttentionCase {
            keys: FloatKind::E4M3,
            ..case
        },
        AttentionCase {
            values: FloatKind::E4M3,
            ..case
        },
        AttentionCase {
            keys: FloatKind::E4M3,
            values: FloatKind::E4M3,
            ..case
        },
    ]
}

/// The wide prompt over every FP8 cache.
pub(crate) const PROMPTS_WIDE_FP8: [AttentionCase; 3] = fp8_caches(PROMPT_WIDE);

/// The wide decode step over every FP8 cache.
pub(crate) const DECODES_WIDE_FP8: [AttentionCase; 3] = fp8_caches(DECODE_WIDE);

/// `case` over a cache quantized per tensor ([`PER_TENSOR`]): `e4m3` keys and values, and `f16`
/// ones, each side under its own scale.
pub(crate) const fn per_tensor_caches(case: AttentionCase) -> [AttentionCase; 2] {
    let scales = Some(PER_TENSOR);
    [
        AttentionCase {
            keys: FloatKind::E4M3,
            values: FloatKind::E4M3,
            scales,
            ..case
        },
        AttentionCase { scales, ..case },
    ]
}

/// The wide prompt over a cache quantized per tensor.
pub(crate) const PROMPTS_WIDE_PER_TENSOR: [AttentionCase; 2] = per_tensor_caches(PROMPT_WIDE);

/// The wide decode step over a cache quantized per tensor.
pub(crate) const DECODES_WIDE_PER_TENSOR: [AttentionCase; 2] = per_tensor_caches(DECODE_WIDE);

/// `case` over a cache in pages of 64 positions, `f16` and `e4m3`, as serving stores it.
pub(crate) const fn paged_caches(case: AttentionCase) -> [AttentionCase; 2] {
    let page_size = NonZeroUsize::new(64);
    let [_, _, fp8] = fp8_caches(case);
    [
        AttentionCase { page_size, ..case },
        AttentionCase { page_size, ..fp8 },
    ]
}

/// The wide prompt over a paged cache.
pub(crate) const PROMPTS_WIDE_PAGED: [AttentionCase; 2] = paged_caches(PROMPT_WIDE);

/// The prompt of head dim 256 over a paged cache.
pub(crate) const PROMPTS_HEAD_DIM_256_PAGED: [AttentionCase; 2] = paged_caches(PROMPT_HEAD_DIM_256);

/// A single position of the same heads, which a routine walks the group of as its queries.
pub(crate) const DECODE: AttentionCase = AttentionCase {
    seq_q: 1,
    visibility: AttentionVisibility::Attended,
    ..PROMPT
};

/// A single position seeing only the last five of its keys: most of the boxes a routine splits the
/// keys into see none of them, and merge nothing.
pub(crate) const DECODE_WINDOWED: AttentionCase = AttentionCase {
    visibility: AttentionVisibility::Recent {
        window: match NonZeroUsize::new(5) {
            Some(window) => window,
            None => unreachable!(),
        },
    },
    ..DECODE
};

/// The prompt with its queries placed first among the keys: query `i` sees the keys up to `i`.
pub(crate) const PROMPT_PLACED_FIRST: AttentionCase = AttentionCase {
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::First,
    },
    ..PROMPT
};

/// The prompt under a mask, on top of its causal diagonal.
pub(crate) const PROMPT_MASKED: AttentionCase = AttentionCase {
    hidden: Some(AttentionCase::stripes),
    ..PROMPT
};

/// The prompt under a mask, its queries placed first.
pub(crate) const PROMPT_MASKED_PLACED_FIRST: AttentionCase = AttentionCase {
    hidden: Some(AttentionCase::stripes),
    ..PROMPT_PLACED_FIRST
};

/// A single position under a mask, its group's heads walked as its queries.
pub(crate) const DECODE_MASKED: AttentionCase = AttentionCase {
    hidden: Some(AttentionCase::stripes),
    ..DECODE
};

/// Every prompt case a routine of many positions is checked on.
pub(crate) const PROMPTS: [AttentionCase; 4] = [
    PROMPT,
    PROMPT_PLACED_FIRST,
    PROMPT_MASKED,
    PROMPT_MASKED_PLACED_FIRST,
];

/// Every decode case a routine of single positions is checked on.
pub(crate) const DECODES: [AttentionCase; 2] = [DECODE, DECODE_MASKED];

/// `case` with both score terms, a sink a row and the short relative bias, over a cache holding
/// eleven positions past its attended keys.
pub(crate) const fn scored(case: AttentionCase) -> AttentionCase {
    AttentionCase {
        sinks: Some(AttentionCase::sink_pattern),
        relative: Some(RELATIVE_SHORT),
        spare: 11,
        ..case
    }
}

/// The prompt under a sliding window, its queries the last positions.
pub(crate) const PROMPT_WINDOWED: AttentionCase = AttentionCase {
    visibility: AttentionVisibility::Windowed {
        window: match NonZeroUsize::new(9) {
            Some(window) => window,
            None => unreachable!(),
        },
        queries: QueryPlacement::Last,
    },
    ..PROMPT
};

/// The prompt with one row of one head hidden from every key: it writes zeros whatever its sink,
/// and its logsumexp is its sink.
pub(crate) const PROMPT_ROW_HIDDEN: AttentionCase = AttentionCase {
    hidden: Some(AttentionCase::one_row),
    sinks: Some(AttentionCase::sink_pattern),
    ..PROMPT
};

/// Every prompt case a routine of many positions is held to under the score terms: a chunk over a
/// partly filled cache, its queries placed first, under a window and under a mask with both terms,
/// each term alone, and a row that sees no key.
pub(crate) const SCORED_PROMPTS: [AttentionCase; 7] = [
    scored(PROMPT),
    scored(PROMPT_PLACED_FIRST),
    scored(PROMPT_WINDOWED),
    scored(PROMPT_MASKED),
    AttentionCase {
        relative: None,
        ..scored(PROMPT)
    },
    AttentionCase {
        sinks: None,
        ..scored(PROMPT)
    },
    PROMPT_ROW_HIDDEN,
];

/// Every decode case a routine of single positions is held to under the score terms: a step over
/// a partly filled cache, through a window, under a mask, and with a relative bias reaching past
/// every key.
pub(crate) const SCORED_DECODES: [AttentionCase; 4] = [
    scored(DECODE),
    scored(DECODE_WINDOWED),
    scored(DECODE_MASKED),
    AttentionCase {
        relative: Some(RELATIVE_LONG),
        ..scored(DECODE)
    },
];

/// The wide prompt and decode step over a paged cache with both score terms: what serving runs.
pub(crate) const SCORED_PAGED: [AttentionCase; 2] = [
    scored(PROMPTS_WIDE_PAGED[0]),
    scored(AttentionCase {
        seq_q: 1,
        visibility: AttentionVisibility::Attended,
        ..PROMPTS_WIDE_PAGED[0]
    }),
];

impl AttentionCase {
    /// A sink that differs from row to row and head to head.
    fn sink_pattern([batch, head, query]: [usize; 3]) -> f32 {
        ((batch + 3 * head + 7 * query) % 11) as f32 / 4. - 1.
    }

    /// A relative bias that differs from row to row, head to head and distance to distance.
    fn relative_pattern([batch, head, query, distance]: [usize; 4]) -> f32 {
        ((batch + 2 * head + 3 * query + 5 * distance) % 13) as f32 / 8. - 0.75
    }

    /// A mask hiding every key from row 2 of head 1.
    fn one_row([_batch, head, query, _key]: [usize; 4]) -> bool {
        head == 1 && query == 2
    }

    /// Positions the cache's buffers hold a sequence: its attended keys and its spare ones.
    fn positions(&self) -> usize {
        self.seq_kv + self.spare
    }

    /// The key position query `i` sits at: placed among the keys by a causal visibility, at the
    /// last attended key otherwise.
    fn query_position(&self, i: usize) -> usize {
        match self.visibility.queries() {
            Some(QueryPlacement::First) => i,
            Some(QueryPlacement::Last) => i + self.seq_kv - self.seq_q,
            None => self.seq_kv - 1,
        }
    }

    fn scale(&self) -> f64 {
        1. / (self.head_dim as f64).sqrt()
    }

    /// A mask that differs from head to head, so a routine reading another head's cells is caught.
    /// It never hides the first key, which a query placed first may be the only one to see.
    fn stripes([_batch, head, query, key]: [usize; 4]) -> bool {
        key > 0 && (query + 2 * key + 3 * head).is_multiple_of(5)
    }

    /// Whether query `i` of head `h` in sequence `b` sees key `j`: the queries sit among the keys
    /// as the visibility places them, the last positions or the first, and the mask hides what it
    /// hides.
    fn sees(&self, b: usize, h: usize, i: usize, j: usize) -> bool {
        let (queries, keys) = (self.seq_q, self.seq_kv);
        let queries_end = match self.visibility.queries() {
            Some(QueryPlacement::First) => queries,
            Some(QueryPlacement::Last) | None => keys,
        };
        let mut seen = true;
        if self.visibility.is_causal() {
            seen &= j + queries <= i + queries_end;
        }
        if let Some(window) = self.visibility.window() {
            seen &= match self.visibility.is_causal() {
                true => j + queries + window.get() > i + queries_end,
                false => j + window.get() >= keys,
            };
        }
        if let Some(hidden) = self.hidden {
            seen &= !hidden([b, h, i, j]);
        }
        seen
    }

    /// The query `[b, h, q, d]`, key `[b, kv, s, d]` and value `[b, kv, s, dv]`, in `[-1, 1)` and
    /// not periodic in any extent a case uses, each rounded to the element it is stored in.
    fn inputs(&self) -> [(Vec<f32>, [usize; 4]); 3] {
        let input = |shape: [usize; 4], salt: usize, element: FloatKind| {
            let values = (0..shape.iter().product::<usize>())
                .map(|i| {
                    let value = ((i * 2654435761 + salt * 40503) % 2048) as f32 / 1024. - 1.;
                    match element {
                        FloatKind::E4M3 => cubecl::e4m3::from_f32(value).to_f32(),
                        _ => f16::from_f32(value).to_f32(),
                    }
                })
                .collect();
            (values, shape)
        };
        [
            input(
                [self.batch, self.heads, self.seq_q, self.head_dim],
                1,
                FloatKind::F16,
            ),
            input(
                [self.batch, self.kv_heads, self.positions(), self.head_dim],
                2,
                self.keys,
            ),
            input(
                [self.batch, self.kv_heads, self.positions(), self.val_dim],
                3,
                self.values,
            ),
        ]
    }

    /// The page of the pool sequence `b` holds its `i`-th page of positions in: the last first.
    fn page(&self, b: usize, i: usize, page_size: NonZeroUsize) -> usize {
        let per_sequence = self.positions().div_ceil(page_size.get());
        self.batch * per_sequence - 1 - (b * per_sequence + i)
    }

    /// `values`, a side of the cache `[b, kv, s, d]`, as a pool of pages `[pages, kv, page_size,
    /// d]`, the positions past the sequences' end holding what no attention may read.
    fn pooled(
        &self,
        values: &[f32],
        [batch, heads, seq, dim]: [usize; 4],
        page_size: NonZeroUsize,
    ) -> (Vec<f32>, [usize; 4]) {
        const UNREAD: f32 = 8.0;
        let size = page_size.get();
        let pages = batch * seq.div_ceil(size);
        let mut pool = vec![UNREAD; pages * heads * size * dim];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    let page = self.page(b, s / size, page_size);
                    let to = ((page * heads + h) * size + s % size) * dim;
                    let from = ((b * heads + h) * seq + s) * dim;
                    pool[to..to + dim].copy_from_slice(&values[from..from + dim]);
                }
            }
        }
        (pool, [pages, heads, size, dim])
    }

    /// The output `[b, q, h, dv]` and the logsumexp `[b, h, q]` on the host, in `f64`, from the
    /// inputs as the device holds them, the keys and values times their side's scale, each seen
    /// score plus its relative bias and each row's softmax weighing its sink at a value of zero,
    /// where the case has them. A query that sees no key writes zeros.
    fn reference(&self) -> [HostData; 2] {
        let Self {
            batch,
            heads,
            kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
            ..
        } = *self;
        let group = heads / kv_heads;
        let positions = self.positions();
        let [(q, _), (k, _), (v, _)] = self.inputs();
        let scale = self.scale();
        let KvScales {
            key: key_scale,
            value: value_scale,
        } = self.scales.unwrap_or(KvScales {
            key: 1.0,
            value: 1.0,
        });
        let (key_scale, value_scale) = (key_scale as f64, value_scale as f64);
        let mut out = vec![0f32; batch * seq_q * heads * val_dim];
        let mut logsumexp = vec![0f32; batch * heads * seq_q];
        for b in 0..batch {
            for h in 0..heads {
                let kv = h / group;
                for i in 0..seq_q {
                    let query = &q[((b * heads + h) * seq_q + i) * head_dim..][..head_dim];
                    let scores: Vec<Option<f64>> = (0..seq_kv)
                        .map(|j| {
                            self.sees(b, h, i, j).then(|| {
                                let key = &k[((b * kv_heads + kv) * positions + j) * head_dim..]
                                    [..head_dim];
                                let distance = self.query_position(i).checked_sub(j);
                                let relative = match (self.relative, distance) {
                                    (Some(RelativeCase { extent, bias }), Some(distance))
                                        if distance < extent =>
                                    {
                                        bias([b, h, i, distance]) as f64
                                    }
                                    _ => 0.0,
                                };
                                query
                                    .iter()
                                    .zip(key)
                                    .map(|(x, y)| *x as f64 * *y as f64 * key_scale)
                                    .sum::<f64>()
                                    * scale
                                    + relative
                            })
                        })
                        .collect();
                    let sink = self.sinks.map(|sink| sink([b, h, i]) as f64);
                    let max = scores
                        .iter()
                        .flatten()
                        .copied()
                        .chain(sink)
                        .reduce(f64::max)
                        .unwrap_or(f64::NEG_INFINITY);
                    let total: f64 = scores
                        .iter()
                        .flatten()
                        .chain(&sink)
                        .map(|s| (s - max).exp())
                        .sum();
                    logsumexp[(b * heads + h) * seq_q + i] = (max + total.ln()) as f32;
                    for c in 0..val_dim {
                        let value: f64 = scores
                            .iter()
                            .enumerate()
                            .filter_map(|(j, s)| s.map(|s| (j, s)))
                            .map(|(j, s)| {
                                (s - max).exp()
                                    * v[((b * kv_heads + kv) * positions + j) * val_dim + c] as f64
                                    * value_scale
                            })
                            .sum();
                        out[((b * seq_q + i) * heads + h) * val_dim + c] = match total > 0.0 {
                            true => (value / total) as f32,
                            false => 0.0,
                        };
                    }
                }
            }
        }
        [
            HostData::row_major(HostDataVec::F32(out), shape![batch, seq_q, heads, val_dim]),
            HostData::row_major(HostDataVec::F32(logsumexp), shape![batch, heads, seq_q]),
        ]
    }

    /// The case's tensors on `client`, asking for each query's logsumexp, its cache under its
    /// scales and the scores under its mask where it has them, the mask stored as `u32` cells.
    fn tensors(&self, client: &Client) -> AttentionTensors {
        let query = f16::elem_type_native();
        let (keys, values) = (ElemType::Float(self.keys), ElemType::Float(self.values));
        let upload = |values: Vec<f32>, shape: Shape, element| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), shape)
                    .dtype(element)
                    .custom(values)
                    .generate_without_host_data(),
            )
        };
        let [(q, q_shape), (k, k_shape), (v, v_shape)] = self.inputs();
        let query = upload(q, Shape::new(q_shape), query);
        let side = |values: Vec<f32>, shape: [usize; 4], element| {
            let (values, shape) = match self.page_size {
                Some(page_size) => self.pooled(&values, shape, page_size),
                None => (values, shape),
            };
            upload(values, Shape::new(shape), element)
        };
        let key = side(k, k_shape, keys);
        let value = side(v, v_shape, values);
        let lengths = upload(vec![self.seq_kv as f32], shape![1], u32::elem_type_native());
        let scale = |scale: f32| upload(vec![scale], shape![1], f32::elem_type_native());
        let scales = match self.scales {
            Some(KvScales { key, value }) => CacheScales {
                key: Some(scale(key)),
                value: Some(scale(value)),
            },
            None => CacheScales::default(),
        };
        let source = KvSource::new(AttentionCache::Float { key, value }).scaled(scales);
        let source = match self.page_size {
            Some(page_size) => {
                let per_sequence = self.positions().div_ceil(page_size.get());
                let table = (0..self.batch)
                    .flat_map(|b| (0..per_sequence).map(move |i| (b, i)))
                    .map(|(b, i)| self.page(b, i, page_size) as f32)
                    .collect();
                source.paged(upload(
                    table,
                    shape![self.batch, per_sequence],
                    u32::elem_type_native(),
                ))
            }
            None => source,
        };
        let tensors = AttentionTensors::attention(
            client,
            AttentionQueries::Dense { query, lengths },
            source,
            AttentionScoring {
                visibility: self.visibility,
                scale: ScoreScale::Custom(self.scale() as f32),
            },
        )
        .expect("the case's tensors are verified")
        .with_logsumexp()
        .expect("a float cache's attention writes its logsumexp");
        let (batch, heads, queries) = (self.batch, self.heads, self.seq_q);
        let rows = || {
            (0..batch).flat_map(move |b| {
                (0..heads).flat_map(move |h| (0..queries).map(move |i| [b, h, i]))
            })
        };
        let tensors = match self.sinks {
            Some(sink) => tensors
                .with_sinks(upload(
                    rows().map(sink).collect(),
                    shape![batch, heads, queries],
                    f32::elem_type_native(),
                ))
                .expect("the sinks are a float a row"),
            None => tensors,
        };
        let tensors = match self.relative {
            Some(RelativeCase { extent, bias }) => tensors
                .with_relative_bias(upload(
                    rows()
                        .flat_map(|[b, h, i]| (0..extent).map(move |d| bias([b, h, i, d])))
                        .collect(),
                    shape![batch, heads, queries, extent],
                    f32::elem_type_native(),
                ))
                .expect("the relative bias is a float a row and distance, over placed queries"),
            None => tensors,
        };
        match self.hidden {
            Some(hidden) => {
                let keys = self.positions();
                let mut cells = Vec::with_capacity(batch * heads * queries * keys);
                for b in 0..batch {
                    for h in 0..heads {
                        for i in 0..queries {
                            for j in 0..keys {
                                cells.push(hidden([b, h, i, j]) as u32 as f32);
                            }
                        }
                    }
                }
                tensors
                    .with_mask(upload(
                        cells,
                        shape![batch, heads, queries, keys],
                        u32::elem_type_native(),
                    ))
                    .expect("the mask is a cell a score")
            }
            None => tensors,
        }
    }
}

/// A routine at one of its settings on one case.
#[derive(Clone)]
pub(crate) struct ForwardCase<S> {
    pub settings: S,
    pub case: AttentionCase,
}

impl<S: core::fmt::Display> Debug for ForwardCase<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {:?}", self.settings, self.case)
    }
}

/// The settings of the routine `L` launches at.
type SettingsOf<L> = <<L as AttentionLaunch>::Routine as Routine>::Settings;

/// An attention against the host's forward.
pub(crate) struct Forward<L> {
    launched: L,
    hw: Hardware,
}

impl<R> Forward<DirectAttention<R>> {
    /// `routine`, which writes the attention, launched alone.
    pub(crate) fn new(routine: R) -> Self {
        Self::of(DirectAttention(routine))
    }
}

impl<R> Forward<SplitKeysAttention<R>> {
    /// The partial routine `partial`, then the split-keys merge.
    pub(crate) fn split_keys(partial: R) -> Self {
        Self::of(SplitKeysAttention { partial })
    }
}

impl<L> Forward<L> {
    fn of(launched: L) -> Self {
        Self {
            launched,
            hw: Hardware::from_client(&cubecl::test_device().client()),
        }
    }
}

impl<L: AttentionLaunch> Forward<L> {
    /// The routine's lead, the first setting it lists on the test device, on each of `cases`;
    /// none where the device lists none.
    pub(crate) fn at_its_lead(&self, cases: &[AttentionCase]) -> Vec<ForwardCase<SettingsOf<L>>> {
        self.at(L::Routine::settings(&self.hw).into_iter().take(1), cases)
    }

    /// Every setting the routine lists on the test device, on each of `cases`.
    pub(crate) fn at_every_setting(
        &self,
        cases: &[AttentionCase],
    ) -> Vec<ForwardCase<SettingsOf<L>>> {
        self.at(L::Routine::settings(&self.hw), cases)
    }

    /// Each of `settings` on each of `cases`.
    pub(crate) fn at(
        &self,
        settings: impl IntoIterator<Item = SettingsOf<L>>,
        cases: &[AttentionCase],
    ) -> Vec<ForwardCase<SettingsOf<L>>> {
        settings
            .into_iter()
            .flat_map(|settings| {
                cases.iter().map(move |&case| ForwardCase {
                    settings: settings.clone(),
                    case,
                })
            })
            .collect()
    }

    /// The settings the routine lists on the test device.
    pub(crate) fn settings(&self) -> Vec<SettingsOf<L>> {
        L::Routine::settings(&self.hw)
    }
}

impl<L: AttentionLaunch> ScopeChecker for Forward<L> {
    type Case = ForwardCase<SettingsOf<L>>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline> {
        let tensors = case.case.tensors(client);
        let logsumexp = tensors.logsumexp().expect("asked for");
        let out = self.launched.launch_on(&self.hw, tensors, &case.settings)?;
        Ok(vec![
            Written {
                name: "out",
                tensor: out,
                host: HostDataType::F32,
            },
            Written {
                name: "logsumexp",
                tensor: logsumexp,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        case.case.reference().into()
    }

    /// `f16` inputs, and a softmax the device's `exp` takes. Over `e4m3` values the probabilities
    /// are cast to `e4m3` beside them, where the device multiplies them on its FP8 instruction:
    /// three bits of mantissa, a relative step of 2⁻⁴ on each weight, which the output carries
    /// times the values, so under the values' scale where they have one.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.case.values {
            FloatKind::E4M3 => {
                let scale = case.case.scales.map_or(1.0, |scales| scales.value.max(1.0));
                2e-2 * scale
            }
            _ => 1e-2,
        }
    }
}
