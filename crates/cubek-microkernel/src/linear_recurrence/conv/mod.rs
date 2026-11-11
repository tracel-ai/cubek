//! The convs of the linear recurrence family: a depthwise causal filter over channels, then, for the
//! q,k,v conv, the L2 norms of the queries and keys. One unit owns a channel: over a window, the
//! write of its newest value overwrites the column the oldest tap reads.

mod causal_operand;
pub mod cube;
mod qk_normed_operand;

pub use causal_operand::*;
pub use qk_normed_operand::*;
