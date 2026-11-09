use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of the conv.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatedDeltaConvOperand {
    /// `[batch, channel]`: the newest value of every channel of the `q|k|v` stream over a window,
    /// or `[batch, stream_position, channel]` over a stream read by tap.
    Qkv,
    /// `[channel, tap]`: the filter of every channel, its newest tap last.
    Filters,
    /// `[store_slot, ring_column, channel]`: the past values of every channel, a ring a slot.
    Window,
    /// `[batch]`: the positions each row has convolved already.
    Lengths,
    /// `[batch, key_head, key_feature]`: the queries, normalized and scaled.
    Query,
    /// `[batch, key_head, key_feature]`: the keys, normalized.
    Key,
    /// `[batch, key_head, group, value_feature]`: the values.
    Value,
}

impl GatedDeltaConvOperand {
    /// The operands a line of channels runs along over a window.
    pub const WINDOW_CHANNELS: [Self; 5] =
        [Self::Qkv, Self::Window, Self::Query, Self::Key, Self::Value];

    /// The operands a line of channels runs along over a stream, which has no window.
    pub const STREAM_CHANNELS: [Self; 4] = [Self::Qkv, Self::Query, Self::Key, Self::Value];

    /// What every microkernel of the conv requires: the `channels` of its case share one width
    /// that divides both `key_features` and `value_features`, and the filters and lengths are read
    /// an element at a time.
    pub fn require(
        channels: &[Self],
        key_features: usize,
        value_features: usize,
        vectorization: &mut Vectorization<Self>,
    ) {
        for &operand in channels {
            for elements in [key_features, value_features] {
                vectorization.require(VectorRequirement::Access { operand, elements });
            }
            vectorization.require(VectorRequirement::Together {
                operand: Self::Qkv,
                with: operand,
            });
        }
        for operand in [Self::Filters, Self::Lengths] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GatedDeltaConvOperandMap<T> {
    pub qkv: T,
    pub filters: T,
    pub window: T,
    pub lengths: T,
    pub query: T,
    pub key: T,
    pub value: T,
}

impl<T: Copy> GatedDeltaConvOperandMap<T> {
    /// The same value for every operand.
    pub fn splat(value: T) -> Self {
        Self {
            qkv: value,
            filters: value,
            window: value,
            lengths: value,
            query: value,
            key: value,
            value,
        }
    }

    /// The value `operand` holds.
    pub fn get(&self, operand: GatedDeltaConvOperand) -> T {
        match operand {
            GatedDeltaConvOperand::Qkv => self.qkv,
            GatedDeltaConvOperand::Filters => self.filters,
            GatedDeltaConvOperand::Window => self.window,
            GatedDeltaConvOperand::Lengths => self.lengths,
            GatedDeltaConvOperand::Query => self.query,
            GatedDeltaConvOperand::Key => self.key,
            GatedDeltaConvOperand::Value => self.value,
        }
    }
}
