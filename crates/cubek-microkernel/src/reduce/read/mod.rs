//! How one unit reads into its carrier: its share of a row that several units read together, every
//! `stride`-th line ([`UnitStridedReduceMicrokernel`]) or the cells a region's walk gives it
//! ([`UnitWalkReduceMicrokernel`]), or the partials several cubes stored of its rows
//! ([`UnitPartialsReduceMicrokernel`]).

mod partials;
mod rows;
mod strided;
mod walk;

pub use partials::*;
pub use rows::*;
pub use strided::*;
pub use walk::*;
