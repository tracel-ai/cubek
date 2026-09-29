//! The kinds of tile ([`base`]): an addressable buffer ([`memory`]), what one plane holds
//! ([`plane`]), and a value computed from its coordinates ([`procedural`]).

pub(crate) mod base;
pub(crate) mod memory;
pub(crate) mod plane;
pub(crate) mod procedural;

pub(crate) use base::*;
pub(crate) use memory::*;
pub(crate) use plane::*;
pub(crate) use procedural::*;
