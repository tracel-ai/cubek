//! What every matmul microkernel computes alone: `out = lhs · rhs`, both row-major.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::zspace::shape;
use cubek_test_utils::{HostData, HostDataVec};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A product on the host: its inputs, small whole numbers exact in every float element a product
/// here reads and sums in, and what it writes.
pub(crate) struct HostMatmul {
    pub lhs: Vec<f32>,
    pub rhs: Vec<f32>,
    shape: [usize; 3],
}

impl HostMatmul {
    /// The inputs of a product of `m`, `n` and `k`.
    pub(crate) fn new(shape: [usize; 3]) -> Self {
        let [m, n, k] = shape;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len).map(|i| (i % modulo) as f32 - offset).collect()
        };
        Self {
            lhs: values(m * k, 7, 3.0),
            rhs: values(k * n, 5, 2.0),
            shape,
        }
    }

    /// `lhs · rhs`, `m × n`.
    pub(crate) fn out(&self) -> HostData {
        let [m, n, k] = self.shape;
        let out = (0..m * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (0..k)
                    .map(|p| self.lhs[i * k + p] * self.rhs[p * n + j])
                    .sum()
            })
            .collect();
        HostData::row_major(HostDataVec::F32(out), shape![m, n])
    }
}
