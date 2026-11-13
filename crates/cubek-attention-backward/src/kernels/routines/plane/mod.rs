//! The plane routines: every plane owns a box of queries of one head, or of keys of one key head,
//! on a matrix instruction, and stages what it walks in shared memory of its own, filled by its
//! own units and met on `sync_plane`, so no plane waits on another. Beside them, the delta
//! routine's planes each write their queries' dot products.

mod base;
mod delta;
mod kernel;
mod key_gradient;
mod query_gradient;

pub use base::*;
pub use delta::*;
pub use key_gradient::*;
pub use query_gradient::*;
