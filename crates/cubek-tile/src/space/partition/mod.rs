//! How a level of the space splits ([`Level`]), and the space with all of its levels
//! ([`Partitioning`]).

mod base;
mod cube_order;
mod distribution;
mod level;
mod levels;
mod quadrant;
mod table;

pub use base::*;
pub use cube_order::*;
pub(crate) use cube_order::{in_plane_axes, swizzled_positions};
pub use distribution::*;
pub(crate) use level::GridCount;
pub use level::*;
pub use levels::*;
pub use quadrant::*;
pub use table::*;
