use crate::attention::{D, DV, S};
use cubecl::tile::Dimension;
use cubek_std::layout::BindingOrder;

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

    /// The lines this operand is bound in, the cache's sides in `order`.
    pub fn lines(self, order: CacheOrder) -> CacheLines {
        match self {
            Self::Key => order.key,
            Self::Value => order.value,
            Self::Query | Self::Out => CacheLines::AsStored,
        }
    }

    /// The dimension this operand is bound innermost along, the cache's sides in `order`.
    pub fn innermost(self, order: CacheOrder) -> Dimension {
        let stored = match self {
            Self::Query | Self::Key => D,
            Self::Value | Self::Out => DV,
        };
        match self.lines(order) {
            CacheLines::AsStored => stored,
            CacheLines::AlongKeys => S,
        }
    }
}

/// Which dimension the lines of one side of the cache run along where it is bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLines {
    /// The one its buffer lies contiguous along: the head dim for keys, the value dim for values.
    AsStored,
    /// The keys, a stride over the same buffer.
    AlongKeys,
}

impl From<BindingOrder> for CacheLines {
    /// The keys' lines where a score reads its rhs in `order`: in their buffer's, as they are
    /// stored, or as the product names it, `[d, s]`.
    fn from(order: BindingOrder) -> Self {
        match order {
            BindingOrder::InBufferOrder => Self::AsStored,
            BindingOrder::Stated => Self::AlongKeys,
        }
    }
}

/// The lines each side of the cache is bound in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheOrder {
    pub key: CacheLines,
    pub value: CacheLines,
}

impl CacheOrder {
    pub const AS_STORED: Self = Self {
        key: CacheLines::AsStored,
        value: CacheLines::AsStored,
    };
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
