//! The depthwise causal conv: a filter over channels, then an optional bias and a silu, over one
//! position of each row or every position of a prompt (a window is a prompt of one position). A
//! row reads the ring columns its length says it holds. A window writes its newest value as the
//! ring's next column in the conv's own launch; a prompt's last valid positions fill the ring in
//! a second launch once every position has read it.
//!
//! The input and the output are `[rows, positions, channels]`, the filters `[channels, taps]`.

mod blueprint;
mod kernel;
mod launch;
mod past;
mod problem;
mod ring_write_back;
mod routine;
mod tensors;

pub use blueprint::*;
pub use launch::*;
pub use past::*;
pub use problem::*;
pub use ring_write_back::RingWriteBackLaunch;
pub use routine::*;
pub use tensors::*;
