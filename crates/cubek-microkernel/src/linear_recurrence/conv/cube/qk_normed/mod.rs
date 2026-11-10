//! The cube scope of the q,k,v conv: a cube holds one key head of one row, or of one position of a
//! prompt, its units dealing the head's channels between them.

mod base;
mod convolved_head;
mod head_channels;
mod microkernel;
mod qk_scale;
mod tiles;

pub use base::*;
pub use microkernel::*;
pub use tiles::*;
