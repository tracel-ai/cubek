//! The logical coordinate space a tile lives in, and how a level of it splits ([`partition`]).

mod base;
mod partition;
mod region;
mod share;

pub use base::*;
pub use partition::*;
pub use region::*;
pub use share::*;
