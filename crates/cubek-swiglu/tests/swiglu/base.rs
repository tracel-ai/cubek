//! What every SwiGLU launch computes: each fused row `[gate | up]` gated into
//! `silu(gate) ⊙ up`, at shapes chosen to be awkward: decode steps of real models' `d_ff`, a
//! prompt's rows, a short half, a prime one that binds one element at a time, one a line past a
//! power of two, and a prime half longer than a group unrolls.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::zspace::shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_swiglu::kernels::SwigluCall;
use cubek_test_utils::{HostData, HostDataVec, TestInput};
use half::f16;

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// Both elements a SwiGLU runs in.
pub(crate) const ELEMENTS: [ElemType; 2] = [F32, F16];

/// `rows` fused rows of `2 · dim`, laid out `[gate | up]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedRows {
    pub rows: usize,
    pub dim: usize,
}

impl FusedRows {
    const fn new(rows: usize, dim: usize) -> Self {
        Self { rows, dim }
    }

    /// Every case a launch is held to.
    pub(crate) fn awkward() -> [Self; 9] {
        [
            // Decode steps: Llama 3.2 1B's `d_ff`, Qwen3-4B's, and a batch of four of a smaller
            // one.
            Self::new(1, 8192),
            Self::new(1, 9728),
            Self::new(4, 2816),
            // A prompt's rows.
            Self::new(37, 1024),
            // A short half, a prime one that binds one element at a time, one a line past a power
            // of two, and one whose lines are a prime count.
            Self::new(3, 96),
            Self::new(2, 111),
            Self::new(2, 1028),
            Self::new(2, 2 * 1031),
            // A prime half longer than a group unrolls: one run, walked in a loop.
            Self::new(2, 5003),
        ]
    }

    /// The rows uploaded as `element`.
    pub(crate) fn call(&self, client: &Client, element: ElemType) -> SwigluCall {
        let values = (0..self.rows * 2 * self.dim).map(Self::wobble).collect();
        let handle = TestInput::builder(client.clone(), shape![self.rows, 2 * self.dim])
            .dtype(element)
            .custom(values)
            .generate_without_host_data();
        SwigluCall {
            client: client.clone(),
            both: PlainTensor::of_handle(handle),
        }
    }

    /// `silu(gate) · up` of each row's halves, both as `element` holds them, in `f64`.
    pub(crate) fn reference(&self, element: ElemType) -> HostData {
        let at = |i: usize| {
            let value = Self::wobble(i);
            match element {
                F16 => f16::from_f32(value).to_f32() as f64,
                _ => value as f64,
            }
        };
        let mut out = Vec::with_capacity(self.rows * self.dim);
        for row in 0..self.rows {
            for d in 0..self.dim {
                let gate = at(row * 2 * self.dim + d);
                let up = at(row * 2 * self.dim + self.dim + d);
                out.push((gate / (1.0 + (-gate).exp()) * up) as f32);
            }
        }
        HostData::row_major(HostDataVec::F32(out), shape![self.rows, self.dim])
    }

    /// The result rounded once to the element.
    pub(crate) fn tolerance(element: ElemType) -> f32 {
        match element {
            F16 => 2e-3,
            _ => 1e-5,
        }
    }

    /// Values in `[-4, 4)`, not periodic in any extent a case uses.
    fn wobble(i: usize) -> f32 {
        ((i.wrapping_mul(2654435761) + 40503) % 2048) as f32 / 256.0 - 4.0
    }
}
