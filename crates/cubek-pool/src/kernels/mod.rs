//! Max pooling as a routine on the tile stack: every output position's window gathered from the
//! input and reduced to its largest tap.

mod geometry;
mod launch;
mod problem;
mod routines;
mod space;

pub use geometry::*;
pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use space::*;
