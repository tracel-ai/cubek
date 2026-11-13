use serde::{Deserialize, Serialize};

/// Where a gate enters the norm: `silu(gate)` multiplies the row before it is normalized or the
/// normalized row after.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateOrder {
    /// `norm(row · silu(gate)) · weight`: the norm sees the gated row.
    GateThenNorm,
    /// `norm(row) · weight · silu(gate)`: the norm sees the row as read, and the gate multiplies
    /// what it writes.
    NormThenGate,
}
