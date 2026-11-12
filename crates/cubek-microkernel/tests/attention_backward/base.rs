//! What every gradient microkernel of attention's backward computes alone, for one sequence of one
//! key head and the group of query heads that read it, each query over the keys a visibility lets
//! it see: with `P = softmax(scale · Q · Kᵀ)` over
//! those keys, zero at the others, read back from the logsumexp, `delta = Σ_dv dO ⊙ O`,
//! `dS = P ⊙ (dO · Vᵀ − delta)`, the gradients `dQ = scale · dS · K`, `dK = scale · dSᵀ · Q` and
//! `dV = Pᵀ · dO`, the key head's summed over every query head of its group.

use crate::attention::{HIDING, HostAttention};
use crate::launch::attention_backward::{
    AttentionBackwardLaunch, AttentionBackwardOperandArgs, AttentionBackwardWrites,
};
use core::fmt::Debug;
use core::marker::PhantomData;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::attention::AttentionVisibility;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
use half::f16;

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The rows of the query and key, and of the value.
const DIMS: [usize; 2] = [32, 32];

/// Query heads a key head in the grouped cases: as many as the widest grouping a microkernel's
/// settings list (four heads a cube), so every setting takes a whole number of them.
const GROUP: usize = 4;

/// Attention's backward on the host over `seq` queries and keys of one key head and `group` query
/// heads, each query over the keys `visibility` lets it see: the forward's operands, small values
/// exact in `f16`, its output rounded to `element` as a forward would write it, the output's
/// gradient, and what each gradient is.
///
/// The query side holds `group · queries` rows, head after head, each row seeing the keys its
/// query does: the key head's gradients sum over every row.
struct HostBackward {
    seq: [usize; 2],
    group: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    out: Vec<f32>,
    logsumexp: Vec<f32>,
    out_gradient: Vec<f32>,
    /// `P` of every row and key, `[g · q, s]`.
    probabilities: Vec<f64>,
}

impl HostBackward {
    fn new(
        seq: [usize; 2],
        group: usize,
        element: ElemType,
        visibility: AttentionVisibility,
    ) -> Self {
        let [queries, keys] = seq;
        let rows = group * queries;
        let [head, val] = DIMS;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len)
                .map(|i| (((i * 37) % modulo) as f32 - offset) / 8.0)
                .collect()
        };
        let (query, key) = (values(rows * head, 13, 6.0), values(keys * head, 11, 5.0));
        let value = values(keys * val, 17, 8.0);
        let out_gradient = values(rows * val, 19, 9.0);
        let scale = 1.0 / (head as f64).sqrt();
        let mut probabilities = vec![0.0f64; rows * keys];
        let mut logsumexp = vec![0.0f32; rows];
        let mut out = vec![0.0f32; rows * val];
        for i in 0..rows {
            // A key the row's query does not see scores minus infinity: its probability is zero.
            let scores: Vec<f64> = (0..keys)
                .map(
                    |j| match HostAttention::sees(seq, visibility, i % queries, j) {
                        true => {
                            scale
                                * (0..head)
                                    .map(|d| query[i * head + d] as f64 * key[j * head + d] as f64)
                                    .sum::<f64>()
                        }
                        false => f64::NEG_INFINITY,
                    },
                )
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            let lse = max + total.ln();
            logsumexp[i] = lse as f32;
            for j in 0..keys {
                probabilities[i * keys + j] = (scores[j] - lse).exp();
            }
            for d in 0..val {
                let o: f64 = (0..keys)
                    .map(|j| probabilities[i * keys + j] * value[j * val + d] as f64)
                    .sum();
                out[i * val + d] = match element {
                    F16 => f16::from_f64(o).to_f32(),
                    _ => o as f32,
                };
            }
        }
        Self {
            seq,
            group,
            query,
            key,
            value,
            out,
            logsumexp,
            out_gradient,
            probabilities,
        }
    }

    /// The scale every score is multiplied by.
    fn scale() -> f32 {
        1.0 / (DIMS[0] as f32).sqrt()
    }

    /// The rows of the query side: every query of every head of the group.
    fn rows(&self) -> usize {
        self.group * self.seq[0]
    }

    /// Every row's `Σ_dv dO ⊙ O`, over the output as the forward wrote it.
    fn delta(&self) -> Vec<f32> {
        let val = DIMS[1];
        (0..self.rows())
            .map(|i| {
                (0..val)
                    .map(|d| self.out_gradient[i * val + d] * self.out[i * val + d])
                    .sum()
            })
            .collect()
    }

    /// `dS` of every row and key, `[g · q, s]`.
    fn score_gradient(&self) -> Vec<f64> {
        let keys = self.seq[1];
        let val = DIMS[1];
        let delta = self.delta();
        (0..self.rows() * keys)
            .map(|at| {
                let (i, j) = (at / keys, at % keys);
                let d_p: f64 = (0..val)
                    .map(|d| self.out_gradient[i * val + d] as f64 * self.value[j * val + d] as f64)
                    .sum();
                self.probabilities[at] * (d_p - delta[i] as f64)
            })
            .collect()
    }

    /// `dQ = scale · dS · K`, `[g, q, d]`, as its microkernel writes it.
    fn query_gradient(&self) -> HostData {
        let [queries, keys] = self.seq;
        let head = DIMS[0];
        let scale = Self::scale() as f64;
        let d_s = self.score_gradient();
        let d_q = (0..self.rows() * head)
            .map(|at| {
                let (i, d) = (at / head, at % head);
                let sum: f64 = (0..keys)
                    .map(|j| d_s[i * keys + j] * self.key[j * head + d] as f64)
                    .sum();
                (scale * sum) as f32
            })
            .collect();
        HostData::row_major(
            HostDataVec::F32(d_q),
            shape![1, 1, self.group, queries, head],
        )
    }

    /// `dK = scale · dSᵀ · Q` and `dV = Pᵀ · dO`, `[s, d]` and `[s, dv]`, summed over every row,
    /// as their microkernel writes them.
    fn key_gradients(&self) -> [HostData; 2] {
        let keys = self.seq[1];
        let rows = self.rows();
        let [head, val] = DIMS;
        let scale = Self::scale() as f64;
        let d_s = self.score_gradient();
        let d_k = (0..keys * head)
            .map(|at| {
                let (j, d) = (at / head, at % head);
                let sum: f64 = (0..rows)
                    .map(|i| d_s[i * keys + j] * self.query[i * head + d] as f64)
                    .sum();
                (scale * sum) as f32
            })
            .collect();
        let d_v = (0..keys * val)
            .map(|at| {
                let (j, d) = (at / val, at % val);
                let sum: f64 = (0..rows)
                    .map(|i| {
                        self.probabilities[i * keys + j] * self.out_gradient[i * val + d] as f64
                    })
                    .sum();
                sum as f32
            })
            .collect();
        [
            HostData::row_major(HostDataVec::F32(d_k), shape![1, 1, keys, head]),
            HostData::row_major(HostDataVec::F32(d_v), shape![1, 1, keys, val]),
        ]
    }

    /// The gradients `writes` names, as their microkernel writes them.
    fn gradients(&self, writes: AttentionBackwardWrites) -> Vec<HostData> {
        match writes {
            AttentionBackwardWrites::Queries => vec![
                self.query_gradient(),
                HostData::row_major(
                    HostDataVec::F32(self.delta()),
                    shape![1, 1, self.group, self.seq[0]],
                ),
            ],
            AttentionBackwardWrites::Keys => self.key_gradients().into(),
            AttentionBackwardWrites::KeysAndQuerySum => {
                let [key_gradient, value_gradient] = self.key_gradients();
                vec![key_gradient, value_gradient, self.query_gradient()]
            }
        }
    }
}

/// A gradient microkernel's settings, and the launch alone they pick on a device.
pub(crate) trait BackwardSettings: Debug + Copy {
    type Microkernel: Microkernel<Settings = Self>;
    type Launch: AttentionBackwardLaunch;

    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<Self::Launch, Refusal>;
}

/// A backward of `element` at `settings`, `steps` of the microkernel's walk long, `group` query
/// heads reading one key head, each query over the keys `visibility` lets it see.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackwardCase<S> {
    pub element: ElemType,
    pub settings: S,
    pub group: usize,
    pub visibility: AttentionVisibility,
    pub steps: usize,
}

impl<S> BackwardCase<S> {
    /// One key head and its group, at the element the forward ran in, summed in `f32`.
    fn item(&self) -> AttentionBackwardWorkItem {
        let [head_dim, val_dim] = DIMS;
        AttentionBackwardWorkItem {
            element: self.element,
            sum: F32,
            group: self.group,
            head_dim,
            val_dim,
        }
    }
}

/// The gradient microkernel `hw` picks at each case's settings.
pub(crate) struct Backward<S> {
    pub hw: Hardware,
    settings: PhantomData<S>,
}

impl<S: BackwardSettings> Backward<S> {
    pub(crate) fn new(hw: Hardware) -> Self {
        Self {
            hw,
            settings: PhantomData,
        }
    }

    /// Every setting the device lists, for a forward in `f16` and in `f32`, one step of the walk
    /// and three, every key attended, one query head a key head; then, at each setting, `f32`
    /// over three steps under each visibility that hides keys ([`HIDING`]) and still shows every
    /// query a key, and over three steps with [`GROUP`] query heads a key head. A walk owned by
    /// the keys holds more queries than keys, so under causal with the queries last its first
    /// queries see none: no attention is defined there.
    pub(crate) fn every_setting(&self) -> Vec<BackwardCase<S>> {
        S::Microkernel::settings(&self.hw)
            .into_iter()
            .flat_map(|settings| {
                let attended = [F16, F32].into_iter().flat_map(move |element| {
                    [1, 3].map(|steps| BackwardCase {
                        element,
                        settings,
                        group: 1,
                        visibility: AttentionVisibility::Attended,
                        steps,
                    })
                });
                let hiding = HIDING
                    .map(|visibility| BackwardCase {
                        element: F32,
                        settings,
                        group: 1,
                        visibility,
                        steps: 3,
                    })
                    .into_iter()
                    .filter(
                        |case| match case.settings.launch(&self.hw, case.item(), case.steps) {
                            Ok(launch) => !HostAttention::hides_every_key_from_a_query(
                                launch.seq(),
                                case.visibility,
                            ),
                            // The refusal shows when the case runs.
                            Err(_) => true,
                        },
                    );
                let grouped = BackwardCase {
                    element: F32,
                    settings,
                    group: GROUP,
                    visibility: AttentionVisibility::Attended,
                    steps: 3,
                };
                attended.chain(hiding).chain([grouped])
            })
            .collect()
    }
}

impl<S: BackwardSettings> ScopeUnderTest for Backward<S> {
    type Case = BackwardCase<S>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = case.settings.launch(&self.hw, case.item(), case.steps)?;
        let [queries, keys] = launch.seq();
        let [head, val] = DIMS;
        let group = case.group;
        let host = HostBackward::new(launch.seq(), group, case.element, case.visibility);
        let delta = host.delta();
        let tensor = |values: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let element = case.element;
        let empty = |len: usize, shape, element| tensor(vec![-7.0; len], shape, element);
        let query_gradient = match launch.writes() {
            // A sum only adds: it arrives holding zero.
            AttentionBackwardWrites::KeysAndQuerySum => tensor(
                vec![0.0; group * queries * head],
                shape![1, 1, group, queries, head],
                F32,
            ),
            _ => empty(
                group * queries * head,
                shape![1, 1, group, queries, head],
                element,
            ),
        };
        let key_gradient = empty(keys * head, shape![1, 1, keys, head], element);
        let value_gradient = empty(keys * val, shape![1, 1, keys, val], element);
        let delta = match launch.writes() {
            AttentionBackwardWrites::Queries => {
                empty(group * queries, shape![1, 1, group, queries], F32)
            }
            _ => tensor(delta, shape![1, 1, group, queries], F32),
        };
        let arg =
            |tensor: &cubecl::std::tensor::TensorHandle| tensor.clone().binding().into_tensor_arg();
        let args = AttentionBackwardOperandArgs {
            query: arg(&tensor(
                host.query,
                shape![1, 1, group, queries, head],
                element,
            )),
            key: arg(&tensor(host.key, shape![1, 1, keys, head], element)),
            value: arg(&tensor(host.value, shape![1, 1, keys, val], element)),
            out: arg(&tensor(
                host.out,
                shape![1, 1, group, queries, val],
                element,
            )),
            out_gradient: arg(&tensor(
                host.out_gradient,
                shape![1, 1, group, queries, val],
                element,
            )),
            logsumexp: arg(&tensor(host.logsumexp, shape![1, 1, group, queries], F32)),
            delta: arg(&delta),
            query_gradient: arg(&query_gradient),
            key_gradient: arg(&key_gradient),
            value_gradient: arg(&value_gradient),
        };
        launch.launch(client, args, HostBackward::scale(), case.visibility)?;
        let written = |name, tensor| Written {
            name,
            tensor,
            host: HostDataType::F32,
        };
        Ok(match launch.writes() {
            AttentionBackwardWrites::Queries => vec![
                written("query_gradient", query_gradient),
                written("delta", delta),
            ],
            AttentionBackwardWrites::Keys => vec![
                written("key_gradient", key_gradient),
                written("value_gradient", value_gradient),
            ],
            AttentionBackwardWrites::KeysAndQuerySum => vec![
                written("key_gradient", key_gradient),
                written("value_gradient", value_gradient),
                written("query_gradient", query_gradient),
            ],
        })
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = case
            .settings
            .launch(&self.hw, case.item(), case.steps)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        HostBackward::new(launch.seq(), case.group, case.element, case.visibility)
            .gradients(launch.writes())
    }

    /// The device's `exp`, and an `f16` forward's rounding of what the gradients read and write.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-3,
            _ => 2e-2,
        }
    }
}
