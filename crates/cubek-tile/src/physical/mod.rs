//! The host-decided physical facts about an operand's bytes: its layout ([`layout`], [`axis`])
//! and the specs, deliveries and builder that bring it into the kernel ([`arg`], [`delivery`],
//! [`source`]). Everything true of an operand before it becomes an in-kernel
//! [`Tile`](crate::Tile) with a [`Space`](crate::Space).
//!
//! Formerly the standalone `cubek-layout` crate plus the `load/` surface, folded together since
//! the tile engine is their only consumer.

mod arg;
mod axis;
mod delivery;
mod layout;
mod source;

pub use arg::*;
pub use axis::*;
pub use delivery::*;
pub use layout::*;
pub use source::*;
