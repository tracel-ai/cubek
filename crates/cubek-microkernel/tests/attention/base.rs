//! What every attention microkernel computes alone: `softmax(scale · q · kᵀ) · v` over one batch
//! and one head, every key attended, all row-major, and each query's `ln Σ exp(scale · q · kᵀ)`.

use core::fmt::Debug;
use core::marker::PhantomData;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::testing::attention::{AttentionElements, AttentionLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

use crate::matmul::{F16, F32};

/// The rows of the query and key, and of the value.
pub(crate) const DIMS: [usize; 2] = [64, 64];

/// An attention on the host: its query, keys and values, small values so the scores stay in a
/// range `f16` holds and the softmax spreads, and what it writes.
pub(crate) struct HostAttention {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    /// The queries and the keys.
    seq: [usize; 2],
    /// The rows of the query and key, and of the value.
    dims: [usize; 2],
}

impl HostAttention {
    /// The inputs of `seq` queries and keys, with rows of `dims` in the query and key and in the
    /// value.
    pub(crate) fn new(seq: [usize; 2], dims: [usize; 2]) -> Self {
        let [queries, keys] = seq;
        let [head, val] = dims;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i % modulo) as f32 - offset) / 4.0)
                .collect()
        };
        Self {
            query: values(queries * head, 7, 3.0),
            key: values(keys * head, 5, 2.0),
            value: values(keys * val, 11, 5.0),
            seq,
            dims,
        }
    }

    /// The scale every score is multiplied by: one over the root of the head.
    pub(crate) fn scale(&self) -> f32 {
        1.0 / (self.dims[0] as f32).sqrt()
    }

    /// The output over `[B, KV, G, Q, DV]`, and the logsumexp over `[B, KV, G, Q]`.
    pub(crate) fn out(&self) -> [HostData; 2] {
        let [queries, keys] = self.seq;
        let [head, val] = self.dims;
        let scale = self.scale();
        let mut out = vec![0.0; queries * val];
        let mut logsumexp = vec![0.0; queries];
        for i in 0..queries {
            let scores: Vec<f32> = (0..keys)
                .map(|j| {
                    scale
                        * (0..head)
                            .map(|d| self.query[i * head + d] * self.key[j * head + d])
                            .sum::<f32>()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            logsumexp[i] = max + total.ln();
            for d in 0..val {
                out[i * val + d] = (0..keys)
                    .map(|j| weights[j] * self.value[j * val + d])
                    .sum::<f32>()
                    / total;
            }
        }
        [
            HostData::row_major(HostDataVec::F32(out), shape![1, 1, 1, queries, val]),
            HostData::row_major(HostDataVec::F32(logsumexp), shape![1, 1, 1, queries]),
        ]
    }
}

/// An attention microkernel's settings, and the launch alone they pick on a device.
pub(crate) trait AttentionSettings: Debug + Copy {
    type Microkernel: Microkernel<Settings = Self>;
    type Launch: AttentionLaunch;

    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        steps: usize,
    ) -> Result<Self::Launch, Refusal>;
}

/// An attention of `elements` at `settings`, `steps` of the microkernel's step of keys long.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionCase<S> {
    pub elements: AttentionElements,
    pub settings: S,
    pub steps: usize,
}

/// The attention microkernel `hw` picks at each case's settings.
pub(crate) struct Attention<S> {
    pub hw: Hardware,
    settings: PhantomData<S>,
}

impl<S: AttentionSettings> Attention<S> {
    pub(crate) fn new(hw: Hardware) -> Self {
        Self {
            hw,
            settings: PhantomData,
        }
    }

    /// Every setting the device lists, for `f16` and `f32` inputs summed in `f32`, one step of
    /// keys and three.
    pub(crate) fn every_setting(&self) -> Vec<AttentionCase<S>> {
        S::Microkernel::settings(&self.hw)
            .into_iter()
            .flat_map(|settings| {
                [F16, F32].into_iter().flat_map(move |input| {
                    let elements = AttentionElements {
                        input,
                        sum: F32,
                        out: F32,
                    };
                    [1, 3].map(|steps| AttentionCase {
                        elements,
                        settings,
                        steps,
                    })
                })
            })
            .collect()
    }
}

impl<S: AttentionSettings> ScopeUnderTest for Attention<S> {
    type Case = AttentionCase<S>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = case.settings.launch(&self.hw, case.elements, case.steps)?;
        let [queries, keys] = launch.seq();
        let [head, val] = DIMS;
        let attention = HostAttention::new(launch.seq(), DIMS);
        let item = launch.item();
        let input = item.elements.query;
        let tensor = |data: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(data)
                .generate_without_host_data()
        };
        let out = tensor(
            vec![-1e4; queries * val],
            shape![1, 1, 1, queries, val],
            item.elements.out,
        );
        let logsumexp = tensor(vec![-1e4; queries], shape![1, 1, 1, queries], item.sum);
        let lengths = tensor(vec![keys as f32], shape![1], ElemType::UInt(UIntKind::U32));
        let scale = attention.scale();
        launch.launch(
            client,
            tensor(attention.query, shape![1, 1, 1, queries, head], input).binding(),
            tensor(attention.key, shape![1, 1, keys, head], input).binding(),
            tensor(attention.value, shape![1, 1, keys, val], input).binding(),
            out.clone().binding(),
            Some(logsumexp.clone().binding()),
            lengths.binding().into_tensor_arg(),
            scale,
        )?;
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
        let launch = case
            .settings
            .launch(&self.hw, case.elements, case.steps)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        HostAttention::new(launch.seq(), DIMS).out().into()
    }

    /// `f16` inputs, and a softmax the device's `exp` takes.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        2e-2
    }
}
