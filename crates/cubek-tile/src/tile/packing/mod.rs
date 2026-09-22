//! How packed values sit in words and decode: the statement ([`base`]) and the view that unpacks
//! a line ([`view`]).

mod base;
mod view;

pub use base::*;
pub(crate) use view::*;
