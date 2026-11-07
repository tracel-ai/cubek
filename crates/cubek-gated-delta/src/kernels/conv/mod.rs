//! The gated delta conv: the depthwise causal filter over the `q|k|v` stream of one position of
//! each row, with the window it reads and writes in place and the L2 norms of the queries and
//! keys. It is the kernel before the walk, and produces the walk's queries, keys and values.
//!
//! Each row reads the columns of its window slot that its length says it holds, and writes its
//! newest values as the window's next column. A column it does not hold reads as zero. The
//! microkernel is [`cubek_microkernel::gated_delta::conv`]'s.

mod blueprint;
mod kernel;
mod problem;
mod routine;
mod tensors;

pub use blueprint::*;
pub use problem::*;
pub use routine::*;
pub use tensors::*;
