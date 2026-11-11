//! The convs of the linear recurrence family: a depthwise causal filter over channels, one
//! position of each row or every position of a prompt, the q,k,v conv adding the L2 norms of the
//! queries and keys. A row reads the ring columns its length says it holds, and a prompt's last
//! valid positions fill the ring in a second launch once every position has read it.

mod blueprint;
mod causal;
#[cfg(test)]
mod fixtures;
mod launch;
mod past;
mod qk_normed;
mod reads;
mod ring_write_back;

pub use blueprint::*;
pub use causal::*;
pub use launch::*;
pub(crate) use past::admit_element;
pub use past::*;
pub use qk_normed::*;
pub use reads::{BoundReads, CausalConvReads};
pub use ring_write_back::RingWriteBack;
