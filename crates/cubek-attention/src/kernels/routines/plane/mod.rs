//! The plane scope: a plane owns a box of rows of one head, from its reads to its writes, its
//! keys and values staged in a window of its own and met on `sync_plane`.

mod base;
mod kernel;

pub use base::*;
