//! The cube scope: a cube holds one key head of one row, or of one position of a stream, its units
//! dealing the head's channels between them.

mod past_taps;
mod qk_normed;

pub use qk_normed::*;
