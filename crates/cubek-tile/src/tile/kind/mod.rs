//! The kinds of tile ([`base`]): an addressable buffer ([`memory`]), what one plane holds
//! ([`plane`]), and a value computed from its coordinates ([`procedural`]).

mod base;
mod memory;
mod plane;
mod procedural;

pub use base::*;
pub use memory::*;
pub use plane::*;
pub use procedural::*;
