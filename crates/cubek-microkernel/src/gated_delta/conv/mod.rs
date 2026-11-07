//! The conv of a gated delta-rule layer: a depthwise causal filter over the `q|k|v` stream of one
//! position of each row, its past kept as a ring of columns in a store, then the L2 norms of the
//! queries and keys.
//!
//! The work item is one key head of one row. One unit owns a channel, since the write of its newest
//! value overwrites the column the oldest tap reads.

pub mod cube;
mod operand;

pub use operand::*;
