//! Partials of the same rows of an attention, each from a worker that walked its own part of the
//! keys, merged into one row: a folder per scope, and the slots the partials lie in, which the
//! attention writes and the merge reads.

pub mod plane;
mod slots;

pub use slots::*;
