//! What every RMS norm microkernel computes alone: each row divided by its root mean square and
//! scaled by the weight, `out = x / sqrt(mean(x²) + ε) · γ`.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::testing::rms_norm::{RmsNormLaunch, RmsNormSettings};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The `ε` every case adds to the mean of squares.
const EPSILON: f32 = 1e-5;

/// Rows of one and several, a width of whole lines, one no wide read divides, and one narrower
/// than a plane's lanes.
pub(crate) const SHAPES: [(usize, usize); 3] = [(1, 1024), (3, 1000), (2, 70)];

/// An RMS norm on the host: its rows and weight, eighths exact in `f16`, and what it writes.
struct HostRmsNorm {
    input: Vec<f32>,
    weight: Vec<f32>,
    rows: usize,
    dim: usize,
}

impl HostRmsNorm {
    fn new(rows: usize, dim: usize) -> Self {
        Self {
            input: (0..rows * dim)
                .map(|i| ((i * 37) % 65) as f32 / 8.0 - 4.0)
                .collect(),
            weight: (0..dim).map(|i| ((i * 11) % 17) as f32 / 8.0).collect(),
            rows,
            dim,
        }
    }

    /// Every row normalized and scaled, `[1, rows, dim]`.
    fn out(&self) -> HostData {
        let dim = self.dim;
        let out = self
            .input
            .chunks(dim)
            .flat_map(|row| {
                let mean = row.iter().map(|x| x * x).sum::<f32>() / dim as f32;
                let factor = 1.0 / (mean + EPSILON).sqrt();
                row.iter()
                    .zip(&self.weight)
                    .map(move |(x, w)| x * factor * w)
            })
            .collect();
        HostData::row_major(HostDataVec::F32(out), shape![1, self.rows, dim])
    }
}

/// The RMS norm microkernel `hw` picks for each case.
pub(crate) struct RmsNorm {
    pub hw: Hardware,
}

/// Rows of `element`, `dim` wide, normalized by the microkernel `settings` pick.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RmsNormCase {
    pub element: ElemType,
    pub rows: usize,
    pub dim: usize,
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
                        element,
                        rows,
                        dim,
                        settings,
                    })
                })
            })
            .collect()
    }
}

impl RmsNorm {
    /// # Errors
    ///
    /// Why the device runs no RMS norm microkernel at `case`.
    fn launch_at(&self, case: &RmsNormCase) -> Result<RmsNormLaunch, Refusal> {
        RmsNormLaunch::new(&self.hw, case.element, case.rows, case.dim, &case.settings)
    }
}

impl ScopeUnderTest for RmsNorm {
    type Case = RmsNormCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let host = HostRmsNorm::new(case.rows, case.dim);
        let tensor = |values: Vec<f32>, shape| {
            TestInput::builder(client.clone(), shape)
                .dtype(case.element)
                .custom(values)
                .generate_without_host_data()
        };
        let input = tensor(host.input, shape![1, case.rows, case.dim]);
        let weight = tensor(host.weight, shape![case.dim]);
        let out = tensor(
            vec![-7.0; case.rows * case.dim],
            shape![1, case.rows, case.dim],
        );
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
            weight.binding().into_tensor_arg(),
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
        vec![HostRmsNorm::new(case.rows, case.dim).out()]
    }

    /// The output's own rounding: `f16` keeps ten bits.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-4,
            _ => 4e-3,
        }
    }
}
