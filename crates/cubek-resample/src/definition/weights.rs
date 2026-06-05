use cubecl::prelude::*;

/// Defines the weights used in the reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// Weights are zero: w_i = 0 for all i. Neutral element for Linear semiring.
    Zero,

    /// Weights are ones: w_i = 1 for all i. Neutral element for Tropical semirings.
    One,

    /// Weights are constant: w_i = c for all i.
    Constant { value: f32 },

    /// Weights are triangle: w_i = i / k for i in 0..k, and w_i = (k - i) / k for i in k..2k.
    Triangle,

    /// Weights are cubic: w_i = (i / k)^3 for i in 0..k, and w_i = ((k - i) / k)^3 for i in k..2k.
    Cubic { a: f32 },

    /// Weights are sinc with window. 'lobes' is usually 2 or 3.
    Lanczos { lobes: u32 },

    /// Weights are given by a tensor.
    Tensor { tensor: TensorBinding<R> },
}
