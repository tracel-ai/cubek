//! The addressable backing store ([`MemData`], gmem and smem): what one is ([`base`]), how it is
//! built ([`of`]) or derived as a shared-memory stage ([`stage`]), how it is touched ([`access`]),
//! and the layouts that address its bytes ([`window`]).

mod access;
mod base;
mod of;
mod stage;
mod window;

pub use base::*;
pub(crate) use stage::*;
pub use window::*;
