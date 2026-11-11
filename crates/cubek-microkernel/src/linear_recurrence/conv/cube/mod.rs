//! The cube scope: a cube's units deal a row's, or a prompt position's, channels between them.
//! What every conv shares is here: the past their taps read, and the filters and the silu that make
//! a line of channels.

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
