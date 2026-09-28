//! The slots a walk fills ahead of the region that reads them: what a slot holds ([`payload`]),
//! when each of its operands is filled ([`plan`]), the slot itself ([`slot`]), the one
//! constructor and fill over them ([`fill`]), and the schedule that drives them ([`base`]).

mod base;
mod fill;
mod payload;
mod plan;
mod slot;

pub use base::*;
pub use payload::pair::OperandPair;
pub use plan::*;
pub use slot::*;
