//! The logical coordinate space a tile lives in, and how a level of it splits ([`partition`]).

mod axis;
mod partition;
mod region;
mod space;

pub use axis::*;
pub use partition::*;
pub use region::*;
pub use space::*;
