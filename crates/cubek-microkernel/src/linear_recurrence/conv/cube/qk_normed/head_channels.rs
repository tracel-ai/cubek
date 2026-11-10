use crate::linear_recurrence::RecurrenceHeads;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Where a key head's channels lie in the `q|k|v` stream: the queries, keys and values come from
/// three ranges of it that no one window of its tile spans.
#[derive(CubeType)]
pub(super) struct HeadChannels {
    head: usize,
    #[cube(comptime)]
    heads: RecurrenceHeads,
}

#[cube]
impl HeadChannels {
    pub fn new(head: usize, #[comptime] heads: RecurrenceHeads) -> Self {
        HeadChannels { head, heads }
    }

    /// The stream channel of the head's channel `local`.
    pub fn first(&self, local: usize) -> usize {
        let (key_features, heads_per_group, value_features) = comptime!((
            self.heads.key_features,
            self.heads.heads_per_group,
            self.heads.value_features
        ));
        let key_dim = comptime!(self.heads.key_dim());
        if local < key_features {
            self.head * key_features + local
        } else if local < 2 * key_features {
            key_dim + self.head * key_features + (local - key_features)
        } else {
            2 * key_dim + self.head * heads_per_group * value_features + (local - 2 * key_features)
        }
    }
}
