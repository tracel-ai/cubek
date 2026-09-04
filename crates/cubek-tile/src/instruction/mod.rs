//! Leaf instructions: one hardware operation, or one fixed instruction sequence with no loop
//! over data.
//!
//! Executed only at leaf tiles, the ones no stated level cuts further. Zero
//! awareness of global spaces, memory stages, or global tile walks: the register loop nests that
//! issue these repeatedly are one layer up, and the leaf dispatch that picks between a hardware
//! leaf and a nest is the verb's, one layer above that.

mod algebra;
mod base;
mod config;
pub mod logsumexp;
mod mma;
pub mod plane;
pub mod registers;

pub use algebra::*;
pub(crate) use base::*;
pub use config::*;
