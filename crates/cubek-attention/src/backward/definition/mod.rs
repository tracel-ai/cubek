//! Configuration types for the FlashAttention backward pass.

/// Tile shape for one of the backward kernels. Q/KV-block sizes; the head
/// dimension is taken from the problem.
///
/// Kept separate from the forward's tiling scheme because the dQ kernel and
/// the dK/dV kernel sweep the two sequence axes in opposite orders and
/// legitimately want different tile shapes.
#[derive(Copy, Clone, Debug)]
pub struct TileShape {
    pub seq_q: usize,
    pub seq_kv: usize,
}

/// Knobs for [`crate::backward::flash_attention_backward`] and the
/// three lower-level kernels.
#[derive(Clone, Debug)]
pub struct BackwardConfig {
    /// Softmax scale `1 / sqrt(d)`. Caller supplies it explicitly so callers
    /// that use non-standard scales (e.g. logit soft-cap) don't have to
    /// fight a default.
    pub scale: f32,

    /// Causal mask. The full mask matrix is never materialized.
    pub causal: bool,

    /// Tile shape for kernel 2 (dQ). `None` defers to the autotuner.
    pub dq_tile: Option<TileShape>,

    /// Tile shape for kernel 3 (dK/dV). `None` defers to the autotuner.
    pub dkdv_tile: Option<TileShape>,
}

impl BackwardConfig {
    /// Defaults: scale derived from `head_dim`, no causal mask, autotuned
    /// tile shapes.
    pub fn from_head_dim(head_dim: usize) -> Self {
        Self {
            scale: (head_dim as f32).sqrt().recip(),
            causal: false,
            dq_tile: None,
            dkdv_tile: None,
        }
    }
}
