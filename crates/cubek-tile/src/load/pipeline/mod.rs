//! The slots a walk fills ahead of the region that reads them.

mod base;
mod fill;
mod slot;

pub use base::*;
pub use slot::*;
// `fill` is slot construction and the fill/consume pairs; nothing else to re-export.
