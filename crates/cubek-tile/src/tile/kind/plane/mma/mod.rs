//! The manual-mma encoding of a plane tile: the fragment and its transports ([`base`]), the
//! block-scaled instruction's loads and execution ([`block_scaled`]), and the registers' widths
//! ([`widths`]).

pub(crate) mod base;
pub(crate) mod block_scaled;
mod widths;

pub(crate) use base::*;
pub(crate) use block_scaled::*;
