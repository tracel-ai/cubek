//! The gated delta conv: the depthwise causal filter over the `q|k|v` stream of one position of
//! each row, or of every position of a stream read by tap, with the L2 norms of the queries and
//! keys. It is the kernel before the walk, and produces the walk's queries, keys and values.
//!
//! Each row reads the columns of its window slot that its length says it holds, and writes its
//! newest values as the window's next column. A column it does not hold reads as zero. A stream
//! reads its own context columns and writes nothing back. The microkernel is
//! [`cubek_microkernel::linear_recurrence::conv`]'s.

mod past;
mod qk_normed;
mod reads;

pub use past::*;
pub use qk_normed::*;
