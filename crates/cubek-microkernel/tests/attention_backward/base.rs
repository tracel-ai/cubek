//! What every gradient microkernel of attention's backward computes alone, for one sequence of one
//! head: with `P = softmax(scale · Q · Kᵀ)` read back from the logsumexp, `delta = Σ_dv dO ⊙ O`,
//! `dS = P ⊙ (dO · Vᵀ − delta)`, the gradients `dQ = scale · dS · K`, `dK = scale · dSᵀ · Q` and
//! `dV = Pᵀ · dO`.

use crate::launch::attention_backward::{
    AttentionBackwardLaunch, AttentionBackwardOperandArgs, AttentionBackwardWrites,
};
use core::fmt::Debug;
use core::marker::PhantomData;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
use half::f16;

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The rows of the query and key, and of the value.
const DIMS: [usize; 2] = [32, 32];

/// Attention's backward on the host over `seq` queries and keys of one head: the forward's
/// operands, small values exact in `f16`, its output rounded to `element` as a forward would
/// write it, the output's gradient, and what each gradient is.
struct HostBackward {
    seq: [usize; 2],
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    out: Vec<f32>,
    logsumexp: Vec<f32>,
    out_gradient: Vec<f32>,
    /// `P` of every query and key, `[q, s]`.
    probabilities: Vec<f64>,
}

impl HostBackward {
    fn new(seq: [usize; 2], element: ElemType) -> Self {
        let [queries, keys] = seq;
        let [head, val] = DIMS;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len)
                .map(|i| (((i * 37) % modulo) as f32 - offset) / 8.0)
                .collect()
        };
        let (query, key) = (
            values(queries * head, 13, 6.0),
            values(keys * head, 11, 5.0),
        );
        let value = values(keys * val, 17, 8.0);
        let out_gradient = values(queries * val, 19, 9.0);
        let scale = 1.0 / (head as f64).sqrt();
        let mut probabilities = vec![0.0f64; queries * keys];
        let mut logsumexp = vec![0.0f32; queries];
        let mut out = vec![0.0f32; queries * val];
        for i in 0..queries {
            let scores: Vec<f64> = (0..keys)
                .map(|j| {
                    scale
                        * (0..head)
                            .map(|d| query[i * head + d] as f64 * key[j * head + d] as f64)
                            .sum::<f64>()
                })
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

    /// Every query's `Σ_dv dO ⊙ O`, over the output as the forward wrote it.
    fn delta(&self) -> Vec<f32> {
        let val = DIMS[1];
        (0..self.seq[0])
            .map(|i| {
                (0..val)
                    .map(|d| self.out_gradient[i * val + d] * self.out[i * val + d])
                    .sum()
            })
            .collect()
    }

    /// `dS` of every query and key, `[q, s]`.
    fn score_gradient(&self) -> Vec<f64> {
        let [queries, keys] = self.seq;
        let val = DIMS[1];
        let delta = self.delta();
        (0..queries * keys)
            .map(|at| {
                let (i, j) = (at / keys, at % keys);
                let d_p: f64 = (0..val)
                    .map(|d| self.out_gradient[i * val + d] as f64 * self.value[j * val + d] as f64)
                    .sum();
                self.probabilities[at] * (d_p - delta[i] as f64)
            })
            .collect()
    }

    /// `dQ = scale · dS · K`, `[q, d]`, as its microkernel writes it.
    fn query_gradient(&self) -> HostData {
        let [queries, keys] = self.seq;
        let head = DIMS[0];
        let scale = Self::scale() as f64;
        let d_s = self.score_gradient();
        let d_q = (0..queries * head)
            .map(|at| {
                let (i, d) = (at / head, at % head);
                let sum: f64 = (0..keys)
                    .map(|j| d_s[i * keys + j] * self.key[j * head + d] as f64)
                    .sum();
                (scale * sum) as f32
            })
            .collect();
        HostData::row_major(HostDataVec::F32(d_q), shape![1, 1, 1, queries, head])
    }

    /// `dK = scale · dSᵀ · Q` and `dV = Pᵀ · dO`, `[s, d]` and `[s, dv]`, as their microkernel
    /// writes them.
    fn key_gradients(&self) -> [HostData; 2] {
        let [queries, keys] = self.seq;
        let [head, val] = DIMS;
        let scale = Self::scale() as f64;
        let d_s = self.score_gradient();
        let d_k = (0..keys * head)
            .map(|at| {
                let (j, d) = (at / head, at % head);
                let sum: f64 = (0..queries)
                    .map(|i| d_s[i * keys + j] * self.query[i * head + d] as f64)
                    .sum();
                (scale * sum) as f32
            })
            .collect();
        let d_v = (0..keys * val)
            .map(|at| {
                let (j, d) = (at / val, at % val);
                let sum: f64 = (0..queries)
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
                HostData::row_major(HostDataVec::F32(self.delta()), shape![1, 1, 1, self.seq[0]]),
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

/// A backward of `element` at `settings`, `steps` of the microkernel's walk long.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackwardCase<S> {
    pub element: ElemType,
    pub settings: S,
    pub steps: usize,
}

impl<S> BackwardCase<S> {
    /// One head, at the element the forward ran in, summed in `f32`.
    fn item(&self) -> AttentionBackwardWorkItem {
        let [head_dim, val_dim] = DIMS;
        AttentionBackwardWorkItem {
            element: self.element,
            sum: F32,
            group: 1,
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
    /// and three.
    pub(crate) fn every_setting(&self) -> Vec<BackwardCase<S>> {
        S::Microkernel::settings(&self.hw)
            .into_iter()
            .flat_map(|settings| {
                [F16, F32].into_iter().flat_map(move |element| {
                    [1, 3].map(|steps| BackwardCase {
                        element,
                        settings,
                        steps,
                    })
                })
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
        let host = HostBackward::new(launch.seq(), case.element);
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
                vec![0.0; queries * head],
                shape![1, 1, 1, queries, head],
                F32,
            ),
            _ => empty(queries * head, shape![1, 1, 1, queries, head], element),
        };
        let key_gradient = empty(keys * head, shape![1, 1, keys, head], element);
        let value_gradient = empty(keys * val, shape![1, 1, keys, val], element);
        let delta = match launch.writes() {
            AttentionBackwardWrites::Queries => empty(queries, shape![1, 1, 1, queries], F32),
            _ => tensor(delta, shape![1, 1, 1, queries], F32),
        };
        let arg =
            |tensor: &cubecl::std::tensor::TensorHandle| tensor.clone().binding().into_tensor_arg();
        let args = AttentionBackwardOperandArgs {
            query: arg(&tensor(host.query, shape![1, 1, 1, queries, head], element)),
            key: arg(&tensor(host.key, shape![1, 1, keys, head], element)),
            value: arg(&tensor(host.value, shape![1, 1, keys, val], element)),
            out: arg(&tensor(host.out, shape![1, 1, 1, queries, val], element)),
            out_gradient: arg(&tensor(
                host.out_gradient,
                shape![1, 1, 1, queries, val],
                element,
            )),
            logsumexp: arg(&tensor(host.logsumexp, shape![1, 1, 1, queries], F32)),
            delta: arg(&delta),
            query_gradient: arg(&query_gradient),
            key_gradient: arg(&key_gradient),
            value_gradient: arg(&value_gradient),
        };
        launch.launch(client, args, HostBackward::scale())?;
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
        HostBackward::new(launch.seq(), case.element).gradients(launch.writes())
    }

    /// The device's `exp`, and an `f16` forward's rounding of what the gradients read and write.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-3,
            _ => 2e-2,
        }
    }
}
