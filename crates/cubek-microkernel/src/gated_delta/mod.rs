//! The gated delta rule's microkernels: the recurrence a gated delta-rule layer carries per value
//! head, walked through a sequence's positions ([`walk`]), and what its scopes share: the axes and
//! the partitioning.

mod layer;
mod partitioning;
pub mod walk;

pub use layer::*;
pub use partitioning::*;
