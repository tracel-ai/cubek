//! What every delta walk microkernel computes alone: each value head's state `S`, `[key_features,
//! value_features]`, at every position decayed by `exp(decay)`, corrected toward the value by the
//! strength, `S += k ⊗ (β · (v - Sᵀk))`, and read by the query, `out = Sᵀq`; the last position's
//! state written back.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::cube::CubeWalkMicrokernelSettings;
use cubek_microkernel::testing::linear_recurrence::walk::WalkLaunch;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A walk on the host: its operands, small values with the keys and queries scaled down so the
/// state stays bounded, and what it writes.
struct HostWalk {
    heads: RecurrenceHeads,
    positions: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    decay: Vec<f32>,
    strength: Vec<f32>,
    state: Vec<f32>,
}

impl HostWalk {
    fn new(heads: RecurrenceHeads, positions: usize) -> Self {
        let value_heads = heads.key_heads * heads.heads_per_group;
        let (kd, vd) = (heads.key_features, heads.value_features);
        let values = |len: usize, modulo: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| (((i * 37) % modulo) as f32 - (modulo / 2) as f32) * scale)
                .collect()
        };
        Self {
            heads,
            positions,
            query: values(positions * heads.key_heads * kd, 13, 1.0 / 32.0),
            key: values(positions * heads.key_heads * kd, 11, 1.0 / 32.0),
            value: values(positions * value_heads * vd, 17, 1.0 / 8.0),
            decay: values(positions * value_heads, 7, 1.0 / 16.0)
                .into_iter()
                .map(|x| x - 0.25)
                .collect(),
            strength: values(positions * value_heads, 9, 1.0 / 16.0)
                .into_iter()
                .map(|x| x + 0.5)
                .collect(),
            state: values(value_heads * kd * vd, 19, 1.0 / 16.0),
        }
    }

    /// The output `[1, t, key_heads, heads_per_group, value_features]`, then the last state
    /// `[1, key_heads, heads_per_group, key_features, value_features]`.
    fn out(&self) -> [HostData; 2] {
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features: kd,
            value_features: vd,
        } = self.heads;
        let (t_len, h_len) = (self.positions, key_heads * heads_per_group);
        let mut out = vec![0.0f64; t_len * h_len * vd];
        let mut last = vec![0.0f64; h_len * kd * vd];
        for h in 0..h_len {
            let kh = h / heads_per_group;
            let at = h * kd * vd;
            let mut s: Vec<f64> = self.state[at..at + kd * vd]
                .iter()
                .map(|&x| x as f64)
                .collect();
            for t in 0..t_len {
                let keyed = (t * key_heads + kh) * kd;
                let k = &self.key[keyed..keyed + kd];
                let q = &self.query[keyed..keyed + kd];
                let scalar = t * h_len + h;
                let factor = (self.decay[scalar] as f64).exp();
                let beta = self.strength[scalar] as f64;
                let v = &self.value[scalar * vd..(scalar + 1) * vd];
                s.iter_mut().for_each(|x| *x *= factor);
                let recalled: Vec<f64> = (0..vd)
                    .map(|j| (0..kd).map(|r| s[r * vd + j] * k[r] as f64).sum())
                    .collect();
                for j in 0..vd {
                    let delta = (v[j] as f64 - recalled[j]) * beta;
                    for r in 0..kd {
                        s[r * vd + j] += k[r] as f64 * delta;
                    }
                    out[scalar * vd + j] = (0..kd).map(|r| s[r * vd + j] * q[r] as f64).sum();
                }
            }
            last[at..at + kd * vd].copy_from_slice(&s);
        }
        let host =
            |values: Vec<f64>| HostDataVec::F32(values.into_iter().map(|x| x as f32).collect());
        [
            HostData::row_major(host(out), shape![1, t_len, key_heads, heads_per_group, vd]),
            HostData::row_major(host(last), shape![1, key_heads, heads_per_group, kd, vd]),
        ]
    }
}

/// The delta walk microkernel `hw` picks for each case.
pub(crate) struct Walk {
    pub hw: Hardware,
}

/// A sequence of `positions` over `heads`, its output in `element`, walked at the microkernel's
/// `settings`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WalkCase {
    pub element: ElemType,
    pub heads: RecurrenceHeads,
    pub positions: usize,
    pub settings: CubeWalkMicrokernelSettings,
}

impl Walk {
    /// # Errors
    ///
    /// Why the device runs no delta walk microkernel at `case`.
    fn launch_at(&self, case: &WalkCase) -> Result<WalkLaunch, Refusal> {
        WalkLaunch::new(&self.hw, case.heads, case.positions, &case.settings)
    }
}

impl ScopeUnderTest for Walk {
    type Case = WalkCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features: kd,
            value_features: vd,
        } = case.heads;
        let t = case.positions;
        let host = HostWalk::new(case.heads, t);
        let tensor = |values: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let element = case.element;
        let query = tensor(host.query, shape![1, t, key_heads, kd], F32);
        let key = tensor(host.key, shape![1, t, key_heads, kd], F32);
        let value = tensor(
            host.value,
            shape![1, t, key_heads, heads_per_group, vd],
            F32,
        );
        let decay = tensor(host.decay, shape![1, t, key_heads, heads_per_group], F32);
        let strength = tensor(host.strength, shape![1, t, key_heads, heads_per_group], F32);
        let state = tensor(
            host.state,
            shape![1, key_heads, heads_per_group, kd, vd],
            F32,
        );
        let out = tensor(
            vec![-7.0; t * key_heads * heads_per_group * vd],
            shape![1, t, key_heads, heads_per_group, vd],
            element,
        );
        launch.launch(
            client,
            query.binding().into_tensor_arg(),
            key.binding().into_tensor_arg(),
            value.binding().into_tensor_arg(),
            decay.binding().into_tensor_arg(),
            strength.binding().into_tensor_arg(),
            state.clone().binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
            element,
        )?;
        Ok(vec![
            Written {
                name: "out",
                tensor: out,
                host: HostDataType::F32,
            },
            Written {
                name: "state",
                tensor: state,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        HostWalk::new(case.heads, case.positions).out().into()
    }

    /// The walk sums in `f32`; an `f16` output rounds what it writes.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-4,
            _ => 4e-3,
        }
    }
}
