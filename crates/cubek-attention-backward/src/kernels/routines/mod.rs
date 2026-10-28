//! The routines of attention's backward pass, one folder per scope: each walks the tiles of
//! queries by keys once owned by the queries, then once owned by the keys, except the hypercube's,
//! whose walk owned by the keys also sums the query's gradient.

mod cube;
mod hypercube;
mod plane;
mod unit;

pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
