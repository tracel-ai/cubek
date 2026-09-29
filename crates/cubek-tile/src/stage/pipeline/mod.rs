//! The slots a walk fills ahead of the region that reads them, and the schedule driving them.

mod base;
mod fill;
mod payload;
mod plan;
mod slot;

pub use base::*;
pub use payload::pair::OperandPair;
pub use payload::triple::OperandTriple;
pub(crate) use plan::*;
pub use slot::*;
