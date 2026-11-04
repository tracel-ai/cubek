//! What makes two walks want the same kernel.

use crate::kernels::walk::{
    GatedDeltaWalkInputs, GatedDeltaWalkProblem, GatedDeltaWalkStateLayout,
};
use core::num::NonZeroUsize;
use cubecl::AutotuneKey;
use cubecl::ir::ElemType;
use cubek_microkernel::gated_delta::GatedDeltaHeads;
use serde::{Deserialize, Serialize};

/// What makes two walks want the same kernel.
///
/// The heads, the output element and the states' layout are exact: a model's config fixes the
/// heads and the element, and the kernel compiles the widths and the count of kept states, which a
/// cache's ring fixes. The sequences and the positions are anchored, being the axes a caller
/// varies. A store's slot count is no part of it: the kernel leaves the slots dynamic and a round
/// measures on a scratch store of its rows.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct GatedDeltaWalkKey {
    heads: GatedDeltaHeads,
    output_element: ElemType,
    state: GatedDeltaWalkStateLayout,
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
        Self::new(
            heads,
            inputs.output_element,
            Self::slotless(inputs.state_layout()),
            batch,
            positions,
        )
    }
}

impl GatedDeltaWalkKey {
    /// `state` with any store's slots dropped to one.
    fn slotless(state: GatedDeltaWalkStateLayout) -> GatedDeltaWalkStateLayout {
        match state {
            GatedDeltaWalkStateLayout::Slots { .. } => GatedDeltaWalkStateLayout::Slots {
                slots: NonZeroUsize::MIN,
            },
            owned @ GatedDeltaWalkStateLayout::Owned { .. } => owned,
        }
    }
}

/// The problem a key describes, at its buckets' sequences and positions: dense operands.
impl From<&GatedDeltaWalkKey> for GatedDeltaWalkProblem {
    fn from(key: &GatedDeltaWalkKey) -> Self {
        Self::dense(
            key.batch,
            key.positions,
            key.heads,
            key.output_element,
            key.state,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store that grows reuses what its smaller sizes tuned, and the owned layout is untouched.
    #[test]
    fn a_stores_slot_count_is_no_part_of_the_key() {
        let slotless = GatedDeltaWalkKey::slotless;
        let slots = |slots| GatedDeltaWalkStateLayout::Slots {
            slots: NonZeroUsize::new(slots).unwrap(),
        };
        assert_eq!(slotless(slots(3)), slotless(slots(65)));
        let owned = GatedDeltaWalkStateLayout::Owned {
            kept: NonZeroUsize::new(3).unwrap(),
        };
        assert_eq!(slotless(owned), owned);
    }
}
