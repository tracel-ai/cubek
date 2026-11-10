//! The cube scope: a cube holds a box of channels of one row, or of one position of a prompt, its
//! units dealing the channels between them. What the two convs share is here: the past their taps
//! read, and the filters and the silu that make a line of channels.

mod box_lines;
mod causal;
mod filters;
mod past_taps;
mod qk_normed;
mod ring_write_back;

pub use box_lines::ChannelBox;
pub use causal::*;
pub use filters::*;
pub use past_taps::*;
pub use qk_normed::*;
pub use ring_write_back::*;
