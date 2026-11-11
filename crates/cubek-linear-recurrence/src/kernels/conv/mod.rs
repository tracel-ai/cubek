//! The q,k,v conv: the depthwise causal filter over the `q|k|v` stream of one position of each row,
//! or of every position of a prompt, with the L2 norms of the queries and keys, producing the
//! walk's queries, keys and values. A row reads the ring columns its length says it holds, and a
//! prompt's last valid positions fill the ring in a second launch once every position has read it.

#[cfg(test)]
mod fixtures;
mod past;
mod qk_normed;
mod reads;
mod ring_write_back;

pub(crate) use past::admit_element;
pub use past::*;
pub use qk_normed::*;
pub use ring_write_back::RingWriteBackLaunch;
