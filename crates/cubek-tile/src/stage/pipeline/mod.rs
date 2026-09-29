//! The slots a walk fills ahead of the region that reads them, and the schedule driving them.

pub(crate) mod base;
pub(crate) mod fill;
pub(crate) mod payload;
pub(crate) mod plan;
pub(crate) mod slot;

pub use base::*;
pub use payload::pair::OperandPair;
pub(crate) use plan::*;
pub use slot::*;
