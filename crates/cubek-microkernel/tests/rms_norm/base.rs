//! What every RMS norm microkernel computes alone: each row divided by its root mean square and
//! scaled by the weight, `out = x / sqrt(mean(x²) + ε) · γ`, gated by `silu(gate)` before the norm
//! or after it.

use crate::launch::RowBatch;
use crate::launch::rms_norm::{RmsNormElements, RmsNormLaunch, RmsNormSettings};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::rms_norm::GateOrder;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The `ε` every case adds to the mean of squares.
const EPSILON: f32 = 1e-5;

/// Rows of one and several, a width of whole lines, one no wide read divides, and one narrower
/// than a plane's lanes.
pub(crate) const SHAPES: [(usize, usize); 3] = [(1, 1024), (3, 1000), (2, 70)];

/// The shape the gated and widened cases run at: several rows, a width no wide read divides.
const GATED_SHAPE: RowBatch = RowBatch { rows: 3, dim: 1000 };

/// `x · sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// An RMS norm on the host: its rows, weight and gate, eighths exact in `f16`, and what it writes.
struct HostRmsNorm {
    input: Vec<f32>,
    weight: Vec<f32>,
    gate: Vec<f32>,
    batch: RowBatch,
}

impl HostRmsNorm {
    fn new(batch: RowBatch) -> Self {
        let RowBatch { rows, dim } = batch;
        Self {
            input: (0..rows * dim)
                .map(|i| ((i * 37) % 65) as f32 / 8.0 - 4.0)
                .collect(),
            weight: (0..dim).map(|i| ((i * 11) % 17) as f32 / 8.0).collect(),
            gate: (0..rows * dim)
                .map(|i| ((i * 23) % 49) as f32 / 8.0 - 3.0)
                .collect(),
            batch,
        }
    }

    /// Every row normalized and scaled, gated as `gate` says, `[1, rows, dim]`.
    fn out(&self, gate: Option<GateOrder>) -> HostData {
        let RowBatch { rows, dim } = self.batch;
        let gates = self.gate.chunks(dim);
        let out = self
            .input
            .chunks(dim)
            .zip(gates)
            .flat_map(|(row, gates)| {
                let row: Vec<f32> = match gate {
                    Some(GateOrder::GateThenNorm) => {
                        row.iter().zip(gates).map(|(x, z)| x * silu(*z)).collect()
                    }
                    Some(GateOrder::NormThenGate) | None => row.to_vec(),
                };
                let mean = row.iter().map(|x| x * x).sum::<f32>() / dim as f32;
                let factor = 1.0 / (mean + EPSILON).sqrt();
                let after = match gate {
                    Some(GateOrder::NormThenGate) => gates.iter().map(|z| silu(*z)).collect(),
                    Some(GateOrder::GateThenNorm) | None => vec![1.0; dim],
                };
                row.into_iter()
                    .zip(&self.weight)
                    .zip(after)
                    .map(move |((x, w), g)| x * factor * w * g)
                    .collect::<Vec<_>>()
            })
            .collect();
        HostData::row_major(HostDataVec::F32(out), shape![1, rows, dim])
    }
}

/// The RMS norm microkernel `hw` picks for each case.
pub(crate) struct RmsNorm {
    pub hw: Hardware,
}

/// Rows read and written at `elements`, `dim` wide, gated as `gate` says and normalized by the
/// microkernel `settings` pick.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RmsNormCase {
    pub elements: RmsNormElements,
    pub gate: Option<GateOrder>,
    pub batch: RowBatch,
    pub settings: RmsNormSettings,
}

impl RmsNormCase {
    /// Every shape in `f16` and `f32` at each of `settings`.
    pub(crate) fn every_shape(settings: impl IntoIterator<Item = RmsNormSettings>) -> Vec<Self> {
        settings
            .into_iter()
            .flat_map(|settings| {
                [F16, F32].into_iter().flat_map(move |element| {
                    SHAPES.map(|(rows, dim)| Self {
                        elements: RmsNormElements {
                            input: element,
                            output: element,
                        },
                        gate: None,
                        batch: RowBatch { rows, dim },
                        settings,
                    })
                })
            })
            .collect()
    }

    /// `f16` rows gated before the norm and after it, written in `f16` and in `f32`, at each of
    /// `settings`: the row held wider than it is read, and the output wider than the input.
    pub(crate) fn gated_and_widened(
        settings: impl IntoIterator<Item = RmsNormSettings>,
    ) -> Vec<Self> {
        let gates = [
            None,
            Some(GateOrder::GateThenNorm),
            Some(GateOrder::NormThenGate),
        ];
        settings
            .into_iter()
            .flat_map(|settings| {
                gates.into_iter().flat_map(move |gate| {
                    [F16, F32].map(|output| Self {
                        elements: RmsNormElements { input: F16, output },
                        gate,
                        batch: GATED_SHAPE,
                        settings,
                    })
                })
            })
            .filter(|case| case.gate.is_some() || case.elements.output == F32)
            .collect()
    }
}

impl RmsNorm {
    /// # Errors
    ///
    /// Why the device runs no RMS norm microkernel at `case`.
    fn launch_at(&self, case: &RmsNormCase) -> Result<RmsNormLaunch, Refusal> {
        RmsNormLaunch::new(
            &self.hw,
            case.elements,
            case.gate,
            case.batch,
            &case.settings,
        )
    }
}

impl ScopeUnderTest for RmsNorm {
    type Case = RmsNormCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let host = HostRmsNorm::new(case.batch);
        let tensor = |values: Vec<f32>, element, shape| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let (rows, dim) = (case.batch.rows, case.batch.dim);
        let input = tensor(host.input, case.elements.input, shape![1, rows, dim]);
        let weight = tensor(host.weight, case.elements.input, shape![dim]);
        let gate = case
            .gate
            .map(|_| tensor(host.gate, case.elements.input, shape![1, rows, dim]));
        let out = tensor(
            vec![-7.0; rows * dim],
            case.elements.output,
            shape![1, rows, dim],
        );
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
            weight.binding().into_tensor_arg(),
            gate.map(|gate| gate.binding().into_tensor_arg()),
            out.clone().binding().into_tensor_arg(),
            EPSILON,
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![HostRmsNorm::new(case.batch).out(case.gate)]
    }

    /// The output's own rounding: `f16` keeps ten bits.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.elements.output {
            F32 => 1e-4,
            _ => 4e-3,
        }
    }
}
