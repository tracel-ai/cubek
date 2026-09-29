//! Packed values in words: the statement ([`base`]) and the unpacking view ([`view`]).

mod base;
mod view;

pub use base::*;
pub(crate) use view::*;
