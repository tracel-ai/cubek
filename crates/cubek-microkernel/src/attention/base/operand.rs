use crate::attention::{D, DV, S};
use crate::matmul::MatmulOrder;
use cubecl::tile::Axis;

/// One of the operands of `out = softmax(query · keyᵀ · scale) · value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionOperand {
    /// `[b, kv, g, q, d]`: a group of query heads beside each key head.
    Query,
    /// `[b, kv, s, d]`, or `[b, kv, d, s]` where a microkernel reads its keys along `s`.
    Key,
    /// `[b, kv, s, dv]`.
    Value,
    /// `[b, kv, g, q, dv]`.
    Out,
}

impl AttentionOperand {
    /// The four operands, in the order this kind names them.
    pub const ALL: [Self; 4] = [Self::Query, Self::Key, Self::Value, Self::Out];

    /// The axis this operand is bound innermost along, the keys in `key_order`: the one its lines
    /// run along.
    pub fn innermost(self, key_order: MatmulOrder) -> Axis {
        match (self, key_order) {
            (Self::Query, _) | (Self::Key, MatmulOrder::InBufferOrder) => D,
            (Self::Key, MatmulOrder::Stated) => S,
            (Self::Value, _) | (Self::Out, _) => DV,
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionOperands<T> {
    pub query: T,
    pub key: T,
    pub value: T,
    pub out: T,
}

impl<T: Copy> AttentionOperands<T> {
    /// The value `operand` holds.
    pub fn get(&self, operand: AttentionOperand) -> T {
        match operand {
            AttentionOperand::Query => self.query,
            AttentionOperand::Key => self.key,
            AttentionOperand::Value => self.value,
            AttentionOperand::Out => self.out,
        }
    }

    /// A value per operand, computed from each operand and its value here.
    pub fn map<U>(&self, f: impl Fn(AttentionOperand, T) -> U) -> AttentionOperands<U> {
        AttentionOperands {
            query: f(AttentionOperand::Query, self.query),
            key: f(AttentionOperand::Key, self.key),
            value: f(AttentionOperand::Value, self.value),
            out: f(AttentionOperand::Out, self.out),
        }
    }
}
