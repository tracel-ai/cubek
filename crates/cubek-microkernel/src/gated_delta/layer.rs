use serde::{Deserialize, Serialize};

/// The heads of a gated delta-rule layer, fixed by the model: every number the kernels' layouts
/// derive from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GatedDeltaHeads {
    pub key_heads: usize,
    /// Value heads reading each key head.
    pub group: usize,
    /// The state's rows: the width of a query and a key.
    pub key_features: usize,
    /// The state's columns: the width of a value and an output.
    pub value_features: usize,
}

impl GatedDeltaHeads {
    /// The value columns a run holds: the largest of 4, 2 and 1 that divides both feature counts, so
    /// a line of the widest read the walk makes never straddles a head. The cap of 4 holds a run's
    /// registers and its shared memory slice to four columns however wide the device reads.
    pub fn columns(&self) -> usize {
        [4, 2, 1]
            .into_iter()
            .find(|columns| {
                self.key_features.is_multiple_of(*columns)
                    && self.value_features.is_multiple_of(*columns)
            })
            .unwrap_or(1)
    }

    /// Value heads, each owning a state.
    pub fn value_heads(&self) -> usize {
        self.key_heads * self.group
    }
}
