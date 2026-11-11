//! The convs of the linear recurrence family: a depthwise causal filter over channels, one
//! position of each row, or every position of a prompt. The q,k,v conv follows its filter with the
//! L2 norms of the queries and keys and is the kernel before the walk; both read the past of their
//! taps through the same ring.
//!
//! Each row reads the columns of its ring that its length says it holds, and a window writes its
//! newest values as the ring's next column. A column it does not hold reads as zero. A prompt reads
//! the ring for the taps before its first position, and a second launch writes the row's last
//! valid positions to the ring once every position has read it. The microkernels are
//! [`cubek_microkernel::linear_recurrence::conv`]'s.

mod causal;
#[cfg(test)]
mod fixtures;
mod past;
mod qk_normed;
mod reads;
mod ring_write_back;

pub use causal::*;
pub(crate) use past::admit_element;
pub use past::*;
pub use qk_normed::*;
pub use ring_write_back::RingWriteBack;
