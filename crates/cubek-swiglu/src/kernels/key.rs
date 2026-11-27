//! What tells two gated activations apart for a caller that measures the routines
//! ([`SwigluKey`]), and the problem a key describes.

use crate::kernels::SwigluRoutineProblem;
use cubecl::ir::ElemType;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_matmul::kernels::tensor::PlainTensor;
use serde::{Deserialize, Serialize};

/// What makes two gated activations want the same kernel.
///
/// The width of a half is **exact**: it is a `d_ff`, fixed by the model config, and the kernel
/// compiles it. The row count is anchored, being the dimension a caller varies.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct SwigluKey {
    elem: ElemType,
    /// The width of a half, which the output takes.
    dim: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    rows: usize,
}

/// The key of the gate of `both`'s fused rows: what tells it apart is their element, width and
/// count.
impl From<&PlainTensor> for SwigluKey {
    fn from(both: &PlainTensor) -> Self {
        let shape = &both.binding.shape;
        let both_halves = shape[shape.len() - 1];
        let rows = shape.iter().product::<usize>() / both_halves;
        Self::new(both.elem, both_halves / 2, rows)
    }
}

/// The problem a tuning key describes, at its bucket's rows: dense fused rows of the key's
/// element.
impl From<&SwigluKey> for SwigluRoutineProblem {
    fn from(key: &SwigluKey) -> Self {
        Self::dense(key.rows, key.dim, key.elem)
    }
}
