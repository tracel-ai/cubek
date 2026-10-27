//! Attention's forward and backward passes on the host, in `f64`, and a case's inputs on the
//! device beside them.

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_attention_backward::{AttentionBackwardGradients, AttentionBackwardTensors};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::AttentionVisibility;
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// The shape of one backward pass, and how its queries see its keys.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
    pub val_dim: usize,
    pub visibility: AttentionVisibility,
}

impl Case {
    pub fn scale(&self) -> f64 {
        1. / (self.head_dim as f64).sqrt()
    }

    /// Whether query `i` sees key `j`: the queries are the last positions of the keys.
    fn sees(&self, i: usize, j: usize) -> bool {
        let (queries, keys) = (self.seq_q, self.seq_kv);
        let mut seen = true;
        if self.visibility.is_causal() {
            seen &= j + queries <= i + keys;
        }
        if let Some(window) = self.visibility.window() {
            seen &= match self.visibility.is_causal() {
                true => j + queries + window.get() > i + keys,
                false => j + window.get() >= keys,
            };
        }
        seen
    }
}

/// A case's inputs on the device, at `element`, and every tensor of both passes on the host.
pub struct Reference {
    pub tensors: AttentionBackwardTensors,
    /// The forward's inputs on the device, `[query, key, value]`, and the loss's gradient against
    /// its output.
    pub inputs: [TensorHandle; 3],
    pub out_gradient: TensorHandle,
    /// `[b, h, q]`.
    pub logsumexp: Vec<f64>,
    /// `[b, h, q, d]`.
    pub query_gradient: Vec<f64>,
    /// `[b, kv, s, d]`.
    pub key_gradient: Vec<f64>,
    /// `[b, kv, s, dv]`.
    pub value_gradient: Vec<f64>,
}

impl Reference {
    /// `case`'s inputs on `client` at `element`, the forward's output and logsumexp computed on the
    /// host from the values the device holds, and the gradients the backward pass must find.
    pub fn new(client: &Client, case: Case, element: ElemType) -> Self {
        let Case {
            batch,
            heads,
            kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
            ..
        } = case;
        let group = heads / kv_heads;
        let wobble =
            |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 1024. - 1.;
        let input = |shape: [usize; 4], salt: usize| {
            let cells = shape.iter().product::<usize>();
            TestInput::builder(client.clone(), Shape::new(shape))
                .dtype(element)
                .custom((0..cells).map(|i| wobble(i, salt)).collect())
                .generate_with_f32_host_data()
        };
        let (query, q) = input([batch, heads, seq_q, head_dim], 1);
        let (key, k) = input([batch, kv_heads, seq_kv, head_dim], 2);
        let (value, v) = input([batch, kv_heads, seq_kv, val_dim], 3);
        let (out_gradient, d_o) = input([batch, heads, seq_q, val_dim], 4);
        let get = |data: &HostData, index: [usize; 4]| data.get_f32(&index) as f64;
        let scale = case.scale();

        let mut out = vec![0f64; batch * heads * seq_q * val_dim];
        let mut logsumexp = vec![0f64; batch * heads * seq_q];
        let mut query_gradient = vec![0f64; batch * heads * seq_q * head_dim];
        let mut key_gradient = vec![0f64; batch * kv_heads * seq_kv * head_dim];
        let mut value_gradient = vec![0f64; batch * kv_heads * seq_kv * val_dim];
        for b in 0..batch {
            for h in 0..heads {
                let kv = h / group;
                for i in 0..seq_q {
                    let row = (b * heads + h) * seq_q + i;
                    let scores: Vec<Option<f64>> = (0..seq_kv)
                        .map(|j| {
                            case.sees(i, j).then(|| {
                                (0..head_dim)
                                    .map(|d| get(&q, [b, h, i, d]) * get(&k, [b, kv, j, d]))
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
                    let lse = max
                        + scores
                            .iter()
                            .flatten()
                            .map(|s| (s - max).exp())
                            .sum::<f64>()
                            .ln();
                    logsumexp[row] = lse;
                    let p: Vec<f64> = scores
                        .iter()
                        .map(|s| s.map_or(0., |s| (s - lse).exp()))
                        .collect();
                    for c in 0..val_dim {
                        out[row * val_dim + c] =
                            (0..seq_kv).map(|j| p[j] * get(&v, [b, kv, j, c])).sum();
                    }
                    let delta: f64 = (0..val_dim)
                        .map(|c| get(&d_o, [b, h, i, c]) * out[row * val_dim + c])
                        .sum();
                    for (j, &probability) in p.iter().enumerate() {
                        let dp: f64 = (0..val_dim)
                            .map(|c| get(&d_o, [b, h, i, c]) * get(&v, [b, kv, j, c]))
                            .sum();
                        let ds = probability * (dp - delta);
                        let key_row = (b * kv_heads + kv) * seq_kv + j;
                        for d in 0..head_dim {
                            query_gradient[row * head_dim + d] +=
                                scale * ds * get(&k, [b, kv, j, d]);
                            key_gradient[key_row * head_dim + d] +=
                                scale * ds * get(&q, [b, h, i, d]);
                        }
                        for c in 0..val_dim {
                            value_gradient[key_row * val_dim + c] +=
                                probability * get(&d_o, [b, h, i, c]);
                        }
                    }
                }
            }
        }
        let device = |shape: Vec<usize>, values: &[f64], element: ElemType| {
            TestInput::builder(client.clone(), Shape::from(shape))
                .dtype(element)
                .custom(values.iter().map(|&x| x as f32).collect())
                .generate_without_host_data()
        };
        let out = device(vec![batch, heads, seq_q, val_dim], &out, element);
        let lse = device(
            vec![batch, heads, seq_q],
            &logsumexp,
            f32::elem_type_native(),
        );
        let plain = |handle: &TensorHandle| PlainTensor::of_handle(handle.clone());
        let tensors = AttentionBackwardTensors::new(
            client,
            plain(&query),
            plain(&key),
            plain(&value),
            plain(&out),
            plain(&out_gradient),
            plain(&lse),
            scale as f32,
            case.visibility,
        )
        .expect("the case's tensors are verified");
        Self {
            tensors,
            inputs: [query, key, value],
            out_gradient,
            logsumexp,
            query_gradient,
            key_gradient,
            value_gradient,
        }
    }

    /// `gradients` against the host's, each value within `tolerance` of it relative to one plus
    /// its size.
    pub fn check(&self, client: &Client, gradients: AttentionBackwardGradients, tolerance: f64) {
        for (name, handle, expected) in [
            ("query_gradient", gradients.query, &self.query_gradient),
            ("key_gradient", gradients.key, &self.key_gradient),
            ("value_gradient", gradients.value, &self.value_gradient),
        ] {
            let actual = HostData::from_tensor_handle(client, handle, HostDataType::F32);
            for (at, &want) in expected.iter().enumerate() {
                let got = actual.data.get_f32(at) as f64;
                assert!(
                    (got - want).abs() <= tolerance * (1. + want.abs()),
                    "{name}[{at}] = {got}, expected {want}"
                );
            }
        }
    }
}
