//! The gated delta conv: the depthwise causal filter over the `q|k|v` stream of one position of
//! each row, or of every position of a prompt, with the L2 norms of the queries and keys. It is the
//! kernel before the walk, and produces the walk's queries, keys and values.
//!
//! Each row reads the columns of its ring that its length says it holds, and a window writes its
//! newest values as the ring's next column. A column it does not hold reads as zero. A prompt reads
//! the ring for the taps before its first position, and a second launch writes the row's last
//! valid positions to the ring once every position has read it. The microkernels are
//! [`cubek_microkernel::linear_recurrence::conv`]'s.

#[cfg(test)]
mod fixtures;
mod past;
mod qk_normed;
mod reads;
mod ring_write_back;

pub(crate) use past::admit_element;
pub use past::*;
pub use qk_normed::*;
pub use ring_write_back::RingWriteBack;
