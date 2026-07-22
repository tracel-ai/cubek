//! How an operand lives across the contraction: a staging slot ([`staging`]) sequenced by a
//! [`pipeline`], driven by [`fill`] (streamed vs pinned). [`resident`] (`promote`) is the
//! register-resident counterpart that brackets the whole operation instead of refilling per region.

mod base;
mod fill;
mod pipeline;
mod resident;

pub use base::*;
pub use pipeline::*;
// fill adds `Staging` fill/consume impls only, resident adds `Tile::promote`; nothing to re-export.
