//! The host-decided physical facts about an operand's bytes: its layout ([`layout`])
//! and the specs, deliveries and builder that bring it into the kernel ([`arg`], [`delivery`],
//! [`source`]). Everything true of an operand before it becomes an in-kernel
//! [`Tile`](crate::Tile) with a [`Space`](crate::Space).
//!
//! Formerly the standalone `cubek-layout` crate plus the `load/` surface, folded together since
//! the tile engine is their only consumer.

mod arg;
mod delivery;
mod geometry;
mod launch;
mod layout;
mod projection;
mod residence;
mod source;

pub use arg::*;
pub use delivery::*;
pub use geometry::*;
pub use launch::*;
pub use layout::*;
pub use projection::*;
pub use residence::*;
pub use source::*;
