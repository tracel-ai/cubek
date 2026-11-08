//! Max and average pooling as routines on the tile stack: every output position's window
//! gathered from the input, then reduced (a max) or contracted against weights (an average).

mod geometry;
mod launch;
mod problem;
mod routines;
mod space;
mod weights;

pub use geometry::*;
pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use space::*;
