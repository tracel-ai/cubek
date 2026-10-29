//! What a backward pass computes.

use cubecl::std::tensor::TensorHandle;

/// The gradients a backward pass computes, each shaped as its input.
#[derive(Debug, Clone)]
pub struct AttentionBackwardGradients {
    /// `[b, h, q, d]`.
    pub query: TensorHandle,
    /// `[b, kv, s, d]`.
    pub key: TensorHandle,
    /// `[b, kv, s, dv]`.
    pub value: TensorHandle,
}
