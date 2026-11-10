//! The convs of the linear recurrence family: a depthwise causal filter over channels, then, for the
//! q,k,v conv, the L2 norms of the queries and keys. The taps' past is a ring of columns in a store:
//! a window takes one position of each row and joins its value to the ring, a prompt takes a run of
//! positions, reads the positions before it by tap and the ring for the taps before its first, and
//! leaves the ring to a launch of its own.
//!
//! One unit owns a channel: over a window, the write of its newest value overwrites the column the
//! oldest tap reads.

pub mod cube;
mod qk_normed_operand;

pub use qk_normed_operand::*;
