//! What every attention routine computes: `out = softmax(scale · q · kᵀ) · v` over the keys each
//! query sees, written `[b, q, h, dv]`, and each query's `ln Σ exp(score)` over them, `[b, h, q]`.
//! The cases are chosen to be awkward: no step of queries or keys divides them, the queries sit
//! among the keys last or first, and a mask differs from head to head.

use core::fmt::Debug;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_attention::kernels::{
    AttentionLaunch, AttentionScoring, AttentionTensors, DirectAttention, ScoreScale,
    SplitKeysAttention,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
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
}

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

/// A single position of the same heads, which a routine walks the group of as its queries.
pub(crate) const DECODE: AttentionCase = AttentionCase {
    seq_q: 1,
    visibility: AttentionVisibility::Attended,
    ..PROMPT
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

/// Every prompt case a routine of many positions is held to.
pub(crate) const PROMPTS: [AttentionCase; 4] = [
    PROMPT,
    PROMPT_PLACED_FIRST,
    PROMPT_MASKED,
    PROMPT_MASKED_PLACED_FIRST,
];

/// Every decode case a routine of single positions is held to.
pub(crate) const DECODES: [AttentionCase; 2] = [DECODE, DECODE_MASKED];

impl AttentionCase {
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
    /// not periodic in any extent a case uses, rounded to `f16`, the element every case runs in.
    fn inputs(&self) -> [(Vec<f32>, [usize; 4]); 3] {
        let input = |shape: [usize; 4], salt: usize| {
            let values = (0..shape.iter().product::<usize>())
                .map(|i| {
                    let value = ((i * 2654435761 + salt * 40503) % 2048) as f32 / 1024. - 1.;
                    f16::from_f32(value).to_f32()
                })
                .collect();
            (values, shape)
        };
        [
            input([self.batch, self.heads, self.seq_q, self.head_dim], 1),
            input([self.batch, self.kv_heads, self.seq_kv, self.head_dim], 2),
            input([self.batch, self.kv_heads, self.seq_kv, self.val_dim], 3),
        ]
    }

    /// The output `[b, q, h, dv]` and the logsumexp `[b, h, q]` on the host, in `f64`, from the
    /// inputs as the device holds them. A query that sees no key writes zeros.
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
        let [(q, _), (k, _), (v, _)] = self.inputs();
        let scale = self.scale();
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
                                let key =
                                    &k[((b * kv_heads + kv) * seq_kv + j) * head_dim..][..head_dim];
                                query
                                    .iter()
                                    .zip(key)
                                    .map(|(x, y)| *x as f64 * *y as f64)
                                    .sum::<f64>()
                                    * scale
                            })
                        })
                        .collect();
                    let max = scores
                        .iter()
                        .flatten()
                        .copied()
                        .reduce(f64::max)
                        .unwrap_or(f64::NEG_INFINITY);
                    let total: f64 = scores.iter().flatten().map(|s| (s - max).exp()).sum();
                    logsumexp[(b * heads + h) * seq_q + i] = (max + total.ln()) as f32;
                    for c in 0..val_dim {
                        let value: f64 = scores
                            .iter()
                            .enumerate()
                            .filter_map(|(j, s)| s.map(|s| (j, s)))
                            .map(|(j, s)| {
                                (s - max).exp()
                                    * v[((b * kv_heads + kv) * seq_kv + j) * val_dim + c] as f64
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

    /// The case's tensors on `client`, asking for each query's logsumexp, under its mask where it
    /// has one, stored as `u32` cells.
    fn tensors(&self, client: &Client) -> AttentionTensors {
        let element = f16::elem_type_native();
        let upload = |values: Vec<f32>, shape: Shape, element| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), shape)
                    .dtype(element)
                    .custom(values)
                    .generate_without_host_data(),
            )
        };
        let [query, key, value] = self
            .inputs()
            .map(|(values, shape)| upload(values, Shape::new(shape), element));
        let lengths = upload(vec![self.seq_kv as f32], shape![1], u32::elem_type_native());
        let tensors = AttentionTensors::new(
            client,
            query,
            key,
            value,
            lengths,
            AttentionScoring {
                visibility: self.visibility,
                scale: ScoreScale::Custom(self.scale() as f32),
            },
        )
        .expect("the case's tensors are verified")
        .with_logsumexp();
        match self.hidden {
            Some(hidden) => {
                let (batch, heads, queries, keys) =
                    (self.batch, self.heads, self.seq_q, self.seq_kv);
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
                tensors.with_mask(upload(
                    cells,
                    shape![batch, heads, queries, keys],
                    u32::elem_type_native(),
                ))
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
        self.at(
            self.launched
                .routine()
                .settings(&self.hw)
                .into_iter()
                .take(1),
            cases,
        )
    }

    /// Every setting the routine lists on the test device, on each of `cases`.
    pub(crate) fn at_every_setting(
        &self,
        cases: &[AttentionCase],
    ) -> Vec<ForwardCase<SettingsOf<L>>> {
        self.at(self.launched.routine().settings(&self.hw), cases)
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
        self.launched.routine().settings(&self.hw)
    }
}

impl<L: AttentionLaunch> ScopeUnderTest for Forward<L> {
    type Case = ForwardCase<SettingsOf<L>>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
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

    /// `f16` inputs, and a softmax the device's `exp` takes.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-2
    }
}
