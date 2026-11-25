//! How one unit reads into its carrier: its share of a row that several units read together, every
//! `stride`-th line ([`UnitStridedReduceMicrokernel`]) or the cells a region's walk gives it
//! ([`UnitWalkReduceMicrokernel`]), or the partials several cubes stored of its rows
//! ([`UnitPartialsReduceMicrokernel`]).

mod line;
mod partials;
mod strided;
mod walk;

pub(crate) use line::accumulate_line;
pub use partials::*;
pub use strided::*;
pub use walk::*;
