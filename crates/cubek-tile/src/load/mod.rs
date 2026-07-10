//! The tile-loading surface: everything a launched operand is made of before it becomes
//! an in-kernel [`Tile`](crate::Tile). [`delivery`] names how an operand's bytes move
//! (strided or TMA); [`arg`] holds the launchable arguments those deliveries ride
//! ([`TileArg`](crate::TileArg) / [`TmaArg`](crate::TmaArg)); [`source`] is the builder
//! every client loads through, so the layout/broadcast wiring lives here, not at each
//! call site.

mod arg;
mod delivery;
mod source;

pub use arg::*;
pub use delivery::*;
pub use source::*;
