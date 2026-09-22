//! The logical coordinate space a tile lives in, and how a level of it splits ([`partition`]).

mod axis;
mod base;
mod coords;
mod partition;
mod region;
mod share;

pub use axis::*;
pub use base::*;
pub use coords::*;
pub use partition::*;
pub use region::*;
pub use share::*;
