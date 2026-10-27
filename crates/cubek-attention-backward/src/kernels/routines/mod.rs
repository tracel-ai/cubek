//! The routines of attention's backward pass, one folder per scope: each walks the tiles of
//! queries by keys once owned by the queries, then once owned by the keys.

mod cube;
mod plane;
mod unit;

pub use cube::*;
pub use plane::*;
pub use unit::*;
