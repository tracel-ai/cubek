//! The depthwise causal conv of the linear recurrence family: a filter over channels, then a bias
//! and a silu. One unit owns a channel: where the conv advances its own ring, the write of its
//! newest value overwrites the column the oldest tap reads.

pub mod cube;
mod operand;

pub use operand::*;
