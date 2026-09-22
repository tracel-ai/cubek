//! The kinds of tile: an addressable buffer ([`memory`]), what one plane holds ([`plane`]), and a
//! value computed from its coordinates ([`procedural`]).

mod memory;
mod plane;
mod procedural;

pub use memory::*;
pub use plane::*;
pub use procedural::*;
