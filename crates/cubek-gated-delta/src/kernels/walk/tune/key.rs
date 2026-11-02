//! What makes two walks want the same kernel.

use crate::kernels::walk::{GatedDeltaWalkInputs, GatedDeltaWalkProblem};
use core::num::NonZeroUsize;
use cubecl::AutotuneKey;
use cubecl::ir::ElemType;
use cubek_microkernel::gated_delta::GatedDeltaHeads;
use serde::{Deserialize, Serialize};

/// What makes two walks want the same kernel.
///
/// The heads, the output element and the count of kept states are exact, since a model's config fixes them and the
/// kernel compiles the widths and the count of kept states, which a cache's ring fixes. The
/// sequences and the positions are anchored, being the axes a caller varies.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct GatedDeltaWalkKey {
    heads: GatedDeltaHeads,
    element: ElemType,
    kept: NonZeroUsize,
    #[autotune(anchor(exp(min = 1, max = 256, base = 2)))]
    batch: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    positions: usize,
}

impl From<&GatedDeltaWalkInputs> for GatedDeltaWalkKey {
    fn from(inputs: &GatedDeltaWalkInputs) -> Self {
        let [batch, positions, key_heads, key_features] = inputs.query.shape().dims::<4>();
        let [_, _, value_heads, value_features] = inputs.value.shape().dims::<4>();
        let heads = GatedDeltaHeads {
            key_heads,
            group: value_heads / key_heads,
            key_features,
            value_features,
        };
        Self::new(heads, inputs.element, inputs.kept, batch, positions)
    }
}

/// The problem a key describes, at its buckets' sequences and positions: dense operands, keeping
/// at most as many states as it has positions.
impl From<&GatedDeltaWalkKey> for GatedDeltaWalkProblem {
    fn from(key: &GatedDeltaWalkKey) -> Self {
        let kept = NonZeroUsize::new(key.kept.get().min(key.positions)).unwrap_or(key.kept);
        Self::dense(key.batch, key.positions, key.heads, key.element, kept)
    }
}
