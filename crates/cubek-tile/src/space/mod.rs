//! The logical coordinate space a tile lives in, and how a level of it splits ([`partition`]).

mod axis;
mod base;
mod coords;
mod extent;
mod matrix;
mod partition;
mod region;
mod runtime;
mod share;
mod walk;

pub use axis::*;
pub use base::*;
pub use coords::*;
pub(crate) use extent::*;
pub use matrix::*;
pub use partition::*;
pub(crate) use region::Step;
pub use region::*;
pub(crate) use runtime::Witness;
pub use share::*;
pub use walk::*;
