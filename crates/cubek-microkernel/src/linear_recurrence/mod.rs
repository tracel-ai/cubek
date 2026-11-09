//! The linear recurrence family's microkernels: the recurrence a layer carries per value
//! head, walked through a sequence's positions ([`walk`]), the conv that produces its queries,
//! keys and values ([`conv`]), and what its scopes share: the axes and the partitioning.

pub mod conv;
mod layer;
mod lengths;
mod partitioning;
pub mod walk;

pub use layer::*;
pub(crate) use lengths::walked;
pub use partitioning::*;
