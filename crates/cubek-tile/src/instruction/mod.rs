//! Leaf instructions: one hardware operation, or one fixed instruction sequence with no loop
//! over data.
//!
//! Executed only at leaf tiles, the ones no stated level cuts further, with no awareness of global
//! spaces, memory stages or tile walks: the register nests that issue these repeatedly are one
//! layer up, and the verb's leaf dispatch between a hardware leaf and a nest is above that.

mod base;
mod config;
pub mod logsumexp;
mod mma;
pub(crate) use mma::rhs_layout;
pub mod registers;

pub use base::*;
pub use config::*;
pub use registers::contract::Side;
