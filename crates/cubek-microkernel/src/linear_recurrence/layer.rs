use serde::{Deserialize, Serialize};

/// The heads of a linear recurrence layer, fixed by the model: every number the kernels' layouts
/// derive from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecurrenceHeads {
    pub key_heads: usize,
    /// Value heads reading each key head. Mamba2's `ngroups` is `key_heads`, not this.
    pub heads_per_group: usize,
    /// The state's rows: the width of a query and a key.
    pub key_features: usize,
    /// The state's columns: the width of a value and an output.
    pub value_features: usize,
}

impl RecurrenceHeads {
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
        self.key_heads * self.heads_per_group
    }

    /// Channels of the `q|k|v` stream the conv mixes.
    pub fn conv_dim(&self) -> usize {
        self.key_heads * self.channels_per_key_head()
    }

    /// Channels before the values: every key head's queries, then keys.
    pub fn key_dim(&self) -> usize {
        self.key_heads * self.key_features
    }

    /// Channels one key head owns: its query and key, and its value heads' values.
    pub fn channels_per_key_head(&self) -> usize {
        2 * self.key_features + self.heads_per_group * self.value_features
    }
}
