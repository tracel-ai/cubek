//! The conv of a gated delta-rule layer: a depthwise causal filter over the `q|k|v` stream, then
//! the L2 norms of the queries and keys. The taps' past is a ring of columns in a store, for one
//! position of each row, or the stream itself read by tap, for the positions of a prompt.
//!
//! The work item is one key head of one row, or of one position of a stream. One unit owns a
//! channel: over a ring, the write of its newest value overwrites the column the oldest tap reads.

pub mod cube;
mod qk_normed_operand;

pub use qk_normed_operand::*;
