//! The logical coordinate space a tile lives in, and how a level of it splits ([`partition`]).

mod axis;
mod base;
mod partition;
mod region;

pub use axis::*;
pub use base::*;
pub use partition::*;
pub use region::*;
