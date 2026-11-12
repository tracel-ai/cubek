//! The depthwise causal conv of the linear recurrence family: a filter over channels, then a bias
//! and a silu. One unit owns a channel: where the conv advances its own ring, the write of its
//! newest value may overwrite the column the oldest tap reads (when the ring has exactly
//! `taps - 1` columns), which the unit read first.

pub mod cube;
mod operand;

pub use operand::*;
