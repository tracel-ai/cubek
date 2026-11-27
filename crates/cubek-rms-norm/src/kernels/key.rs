//! What tells two norms apart for a caller that measures the routines ([`RmsNormKey`]), and the
//! problem a key describes.

use crate::kernels::RmsNormRoutineProblem;
use cubecl::ir::ElemType;
use cubecl::{self as cubecl, AutotuneKeyable};
use cubek_matmul::kernels::tensor::PlainTensor;
use serde::{Deserialize, Serialize};

/// What makes two norms want the same kernel.
///
/// The row width is **exact**: it is a `d_model` or a `head_dim`, fixed by the model config, and
/// every owner sizes its registers by it. The row count is anchored — it is the dimension a caller
/// varies freely (batch × sequence) and the one that decides whether the grid is full.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKeyable)]
pub struct RmsNormKey {
    elem: ElemType,
    /// The normalized row's width.
    dim: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    rows: usize,
}

/// The key of the norm of `x`'s rows: what tells it apart is the rows' element, width and count.
impl From<&PlainTensor> for RmsNormKey {
    fn from(x: &PlainTensor) -> Self {
        let shape = &x.binding.shape;
        let d = shape[shape.len() - 1];
        let rows = shape.iter().product::<usize>() / d;
        Self::new(x.elem, d, rows)
    }
}

/// The problem a tuning key describes, at its bucket's rows: dense operands of the key's element.
impl From<&RmsNormKey> for RmsNormRoutineProblem {
    fn from(key: &RmsNormKey) -> Self {
        Self::dense(key.rows, key.dim, key.elem)
    }
}
