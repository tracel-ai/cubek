//! Partials of the same rows of an attention, each from a worker that walked its own part of the
//! keys, merged into one row: the slots the partials lie in, which the attention writes and the
//! merge reads, and a folder per scope.

mod base;
pub mod hypercube;
pub mod plane;

pub use base::*;
