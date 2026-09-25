//! The addressable backing store ([`MemData`], gmem and smem): what one is ([`base`]), how it is
//! built ([`of`]) or derived as a shared-memory stage ([`stage`]), how it is touched ([`access`]),
//! the layouts that address its bytes ([`window`]), and the origin and strides a reader addresses
//! a window from itself ([`address`]).

mod access;
mod address;
mod base;
mod of;
mod stage;
mod window;

pub use access::{MOST_FETCHED_SCALARS, fetched_scalars};
pub use address::WindowAddress;
pub use base::*;
pub(crate) use stage::*;
pub use window::*;
