//! How a level of the space splits ([`Level`]), and the space with all of its levels
//! ([`Partitioning`]).

pub(crate) mod base;
pub(crate) mod cube_order;
pub(crate) mod level;
pub(crate) mod levels;
pub(crate) mod table;

pub use base::*;
pub use cube_order::*;
pub(crate) use cube_order::{in_plane_axes, swizzled_positions};
pub(crate) use level::GridCount;
pub use level::*;
pub use table::*;
