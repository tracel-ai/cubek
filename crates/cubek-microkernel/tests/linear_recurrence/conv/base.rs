//! What every q|k-normed conv microkernel computes alone over a stream: each channel at position
//! `t` the silu of `Σ_j filters[c, j] · qkv[t + j, c]`, rounded to the stream's element before and
//! after as the composed ops hold it; every key head's queries then divided by their L2 norm and
//! the root of their width, its keys by their L2 norm.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::testing::linear_recurrence::conv::ConvLaunch;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
use half::f16;

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The `ε` every norm adds to its sum of squares.
const EPSILON: f32 = 1e-6;

/// A conv on the host: its stream and filters, small values exact in `f16`, and what it writes.
struct HostConv {
    launch: ConvLaunch,
    element: ElemType,
    qkv: Vec<f32>,
    filters: Vec<f32>,
}

impl HostConv {
    fn new(launch: ConvLaunch, element: ElemType) -> Self {
        let conv_dim = launch.heads.conv_dim();
        Self {
            qkv: (0..launch.stream_len() * conv_dim)
                .map(|i| ((i * 37) % 23) as f32 / 8.0 - 1.375)
                .collect(),
            filters: (0..conv_dim * launch.taps)
                .map(|i| ((i * 11) % 13) as f32 / 16.0 - 0.375)
                .collect(),
            launch,
            element,
        }
    }

    /// `value` rounded to the stream's element.
    fn rounded(&self, value: f32) -> f32 {
        match self.element {
            F16 => f16::from_f32(value).to_f32(),
            _ => value,
        }
    }

    /// The queries and keys `[1, t, key_heads, key_features]`, then the values
    /// `[1, t, key_heads, heads_per_group, value_features]`.
    fn out(&self) -> [HostData; 3] {
        let launch = &self.launch;
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features: kd,
            value_features: vd,
        } = launch.heads;
        let (conv_dim, taps, t_len) = (launch.heads.conv_dim(), launch.taps, launch.positions);
        let key_dim = launch.heads.key_dim();
        let mut query = Vec::with_capacity(t_len * key_dim);
        let mut key = Vec::with_capacity(t_len * key_dim);
        let mut value = Vec::with_capacity(t_len * (conv_dim - 2 * key_dim));
        for t in 0..t_len {
            let convolved: Vec<f32> = (0..conv_dim)
                .map(|c| {
                    let sum: f32 = (0..taps)
                        .map(|j| self.filters[c * taps + j] * self.qkv[(t + j) * conv_dim + c])
                        .sum();
                    let mixed = self.rounded(sum);
                    self.rounded(mixed / (1.0 + (-mixed).exp()))
                })
                .collect();
            for head in 0..key_heads {
                let normed = |first: usize, scale: f32| -> Vec<f32> {
                    let line = &convolved[first..first + kd];
                    let squares: f32 = line.iter().map(|x| x * x).sum();
                    let factor = 1.0 / ((squares + EPSILON).sqrt() * scale);
                    line.iter().map(|x| x * factor).collect()
                };
                query.extend(normed(head * kd, (kd as f32).sqrt()));
                key.extend(normed(key_dim + head * kd, 1.0));
            }
            value.extend_from_slice(&convolved[2 * key_dim..]);
        }
        let host = |values: Vec<f32>, shape| HostData::row_major(HostDataVec::F32(values), shape);
        [
            host(query, shape![1, t_len, key_heads, kd]),
            host(key, shape![1, t_len, key_heads, kd]),
            host(value, shape![1, t_len, key_heads, heads_per_group, vd]),
        ]
    }
}

/// The conv microkernel `hw` picks for each case.
pub(crate) struct Conv {
    pub hw: Hardware,
}

/// A stream of `element` over `heads`, `positions` long, convolved with filters of `taps`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvCase {
    pub element: ElemType,
    pub heads: RecurrenceHeads,
    pub taps: usize,
    pub positions: usize,
}

impl Conv {
    /// # Errors
    ///
    /// Why the device runs no conv microkernel at `case`.
    fn launch_at(&self, case: &ConvCase) -> Result<ConvLaunch, Refusal> {
        ConvLaunch::new(&self.hw, case.heads, case.taps, case.positions)
    }
}

impl ScopeUnderTest for Conv {
    type Case = ConvCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features: kd,
            value_features: vd,
        } = case.heads;
        let (t, conv_dim) = (case.positions, case.heads.conv_dim());
        let host = HostConv::new(launch, case.element);
        let tensor = |values: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let qkv = tensor(
            host.qkv,
            shape![1, launch.stream_len(), conv_dim],
            case.element,
        );
        let filters = tensor(host.filters, shape![conv_dim, case.taps], case.element);
        let keyed = || {
            tensor(
                vec![-7.0; t * key_heads * kd],
                shape![1, t, key_heads, kd],
                F32,
            )
        };
        let (query, key) = (keyed(), keyed());
        let value = tensor(
            vec![-7.0; t * key_heads * heads_per_group * vd],
            shape![1, t, key_heads, heads_per_group, vd],
            F32,
        );
        launch.launch(
            client,
            qkv.binding().into_tensor_arg(),
            filters.binding().into_tensor_arg(),
            query.clone().binding().into_tensor_arg(),
            key.clone().binding().into_tensor_arg(),
            value.clone().binding().into_tensor_arg(),
            case.element,
            EPSILON,
        )?;
        let written = |name, tensor| Written {
            name,
            tensor,
            host: HostDataType::F32,
        };
        Ok(vec![
            written("query", query),
            written("key", key),
            written("value", value),
        ])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        HostConv::new(launch, case.element).out().into()
    }

    /// The device's `exp` and an `f16` stream's rounding of a sum taken in another order.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-4,
            _ => 4e-3,
        }
    }
}
