use crate::attention::{D, DV, Q};
use cubecl::tile::Axis;

/// One of the tensors of attention's backward pass: the forward's operands and its output, the
/// output's gradient, what each query's softmax leaves, and the three gradients it computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionBackwardOperand {
    /// `[b, kv, g, q, d]`.
    Query,
    /// `[b, kv, s, d]`.
    Key,
    /// `[b, kv, s, dv]`.
    Value,
    /// `[b, kv, g, q, dv]`: the forward's output.
    Out,
    /// `[b, kv, g, q, dv]`: the loss's gradient against the output.
    OutGradient,
    /// `[b, kv, g, q]`: each query's logsumexp of its scaled scores, which the forward leaves.
    Logsumexp,
    /// `[b, kv, g, q]`: each query's `Σ_dv out_gradient · out`, which the query gradient's walk
    /// writes and the key gradient's reads.
    Delta,
    /// `[b, kv, g, q, d]`.
    QueryGradient,
    /// `[b, kv, s, d]`.
    KeyGradient,
    /// `[b, kv, s, dv]`.
    ValueGradient,
}

impl AttentionBackwardOperand {
    /// Every operand, in the order this kind names them.
    pub const ALL: [Self; 10] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::Out,
        Self::OutGradient,
        Self::Logsumexp,
        Self::Delta,
        Self::QueryGradient,
        Self::KeyGradient,
        Self::ValueGradient,
    ];

    /// The axis this operand is bound innermost along: the one its lines run along.
    pub fn innermost(self) -> Axis {
        match self {
            Self::Query | Self::Key | Self::QueryGradient | Self::KeyGradient => D,
            Self::Value | Self::Out | Self::OutGradient | Self::ValueGradient => DV,
            Self::Logsumexp | Self::Delta => Q,
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionBackwardOperands<T> {
    pub query: T,
    pub key: T,
    pub value: T,
    pub out: T,
    pub out_gradient: T,
    pub logsumexp: T,
    pub delta: T,
    pub query_gradient: T,
    pub key_gradient: T,
    pub value_gradient: T,
}

impl<T: Copy> AttentionBackwardOperands<T> {
    /// The value `operand` holds.
    pub fn get(&self, operand: AttentionBackwardOperand) -> T {
        match operand {
            AttentionBackwardOperand::Query => self.query,
            AttentionBackwardOperand::Key => self.key,
            AttentionBackwardOperand::Value => self.value,
            AttentionBackwardOperand::Out => self.out,
            AttentionBackwardOperand::OutGradient => self.out_gradient,
            AttentionBackwardOperand::Logsumexp => self.logsumexp,
            AttentionBackwardOperand::Delta => self.delta,
            AttentionBackwardOperand::QueryGradient => self.query_gradient,
            AttentionBackwardOperand::KeyGradient => self.key_gradient,
            AttentionBackwardOperand::ValueGradient => self.value_gradient,
        }
    }

    /// A value per operand, computed from each operand and its value here.
    pub fn map<U>(
        &self,
        f: impl Fn(AttentionBackwardOperand, T) -> U,
    ) -> AttentionBackwardOperands<U> {
        use AttentionBackwardOperand as Operand;
        AttentionBackwardOperands {
            query: f(Operand::Query, self.query),
            key: f(Operand::Key, self.key),
            value: f(Operand::Value, self.value),
            out: f(Operand::Out, self.out),
            out_gradient: f(Operand::OutGradient, self.out_gradient),
            logsumexp: f(Operand::Logsumexp, self.logsumexp),
            delta: f(Operand::Delta, self.delta),
            query_gradient: f(Operand::QueryGradient, self.query_gradient),
            key_gradient: f(Operand::KeyGradient, self.key_gradient),
            value_gradient: f(Operand::ValueGradient, self.value_gradient),
        }
    }
}
