//! What every SwiGLU microkernel computes alone: each fused row `[gate | up]` gated into
//! `silu(gate) ⊙ up`.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::swiglu::cube::CubeSwigluMicrokernelSettings;
use cubek_microkernel::testing::swiglu::SwigluLaunch;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A SwiGLU on the host: its fused rows, eighths between `-4` and `4`, exact in `f16`, and what it
/// writes.
struct HostSwiglu {
    input: Vec<f32>,
    rows: usize,
    dim: usize,
}

impl HostSwiglu {
    fn new(rows: usize, dim: usize) -> Self {
        let input = (0..rows * 2 * dim)
            .map(|i| ((i * 37) % 65) as f32 / 8.0 - 4.0)
            .collect();
        Self { input, rows, dim }
    }

    /// `silu(gate) · up`, `[rows, dim]`.
    fn out(&self) -> HostData {
        let dim = self.dim;
        let out = (0..self.rows * dim)
            .map(|idx| {
                let (r, d) = (idx / dim, idx % dim);
                let gate = self.input[r * 2 * dim + d];
                let up = self.input[r * 2 * dim + dim + d];
                gate / (1.0 + (-gate).exp()) * up
            })
            .collect();
        HostData::row_major(HostDataVec::F32(out), shape![self.rows, dim])
    }
}

/// The SwiGLU microkernel `hw` picks for each case.
pub(crate) struct Swiglu {
    pub hw: Hardware,
}

/// Rows of `element`, each half `dim` wide, gated at the microkernel's `settings`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SwigluCase {
    pub element: ElemType,
    pub rows: usize,
    pub dim: usize,
    pub settings: CubeSwigluMicrokernelSettings,
}

impl Swiglu {
    /// # Errors
    ///
    /// Why the device runs no SwiGLU microkernel at `case`.
    fn launch_at(&self, case: &SwigluCase) -> Result<SwigluLaunch, Refusal> {
        SwigluLaunch::new(&self.hw, case.element, case.rows, case.dim, &case.settings)
    }
}

impl ScopeUnderTest for Swiglu {
    type Case = SwigluCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let host = HostSwiglu::new(case.rows, case.dim);
        let input = TestInput::builder(client.clone(), shape![case.rows, 2, case.dim])
            .dtype(case.element)
            .custom(host.input)
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), shape![case.rows, case.dim])
            .dtype(case.element)
            .custom(vec![-7.0; case.rows * case.dim])
            .generate_without_host_data();
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![HostSwiglu::new(case.rows, case.dim).out()]
    }

    /// The output's own rounding: `f16` keeps ten bits.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-5,
            _ => 2e-3,
        }
    }
}
