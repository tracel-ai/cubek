use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of the conv.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QkNormedConvOperand {
    /// `[batch, channel]`: the newest value of every channel of the `q|k|v` stream over a window,
    /// or `[batch, stream_position, channel]` over a prompt.
    Qkv,
    /// `[channel, tap]`: the filter of every channel, its newest tap last.
    Filters,
    /// `[store_slot, ring_column, channel]`: the past values of every channel, a ring a slot.
    Ring,
    /// `[batch]`: the positions each row has convolved already.
    Lengths,
    /// `[batch, key_head, key_feature]`: the queries, normalized and scaled.
    Query,
    /// `[batch, key_head, key_feature]`: the keys, normalized.
    Key,
    /// `[batch, key_head, head_in_group, value_feature]`: the values.
    Value,
}

impl QkNormedConvOperand {
    /// The operands a line of channels runs along.
    pub const CHANNELS: [Self; 5] = [Self::Qkv, Self::Ring, Self::Query, Self::Key, Self::Value];

    /// What every microkernel of the conv requires: the operands along the channels share one width
    /// that divides both `key_features` and `value_features`, and the filters and the lengths are
    /// read an element at a time.
    pub fn require(
        key_features: usize,
        value_features: usize,
        vectorization: &mut Vectorization<Self>,
    ) {
        for operand in Self::CHANNELS {
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
pub struct QkNormedConvOperandMap<T> {
    pub qkv: T,
    pub filters: T,
    pub ring: T,
    pub lengths: T,
    pub query: T,
    pub key: T,
    pub value: T,
}

impl<T: Copy> QkNormedConvOperandMap<T> {
    /// The same value for every operand.
    pub fn splat(value: T) -> Self {
        Self {
            qkv: value,
            filters: value,
            ring: value,
            lengths: value,
            query: value,
            key: value,
            value,
        }
    }

    /// The value `operand` holds.
    pub fn get(&self, operand: QkNormedConvOperand) -> T {
        match operand {
            QkNormedConvOperand::Qkv => self.qkv,
            QkNormedConvOperand::Filters => self.filters,
            QkNormedConvOperand::Ring => self.ring,
            QkNormedConvOperand::Lengths => self.lengths,
            QkNormedConvOperand::Query => self.query,
            QkNormedConvOperand::Key => self.key,
            QkNormedConvOperand::Value => self.value,
        }
    }
}
