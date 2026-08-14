//! How an operand lives across the contraction, i.e. what its [`Residence`](crate::Residence)
//! costs: a staging slot ([`base`]) sequenced by a [`pipeline`], driven by [`fill`] (streamed vs
//! pinned, materialized vs rebound). [`resident`] (`promote`) is the output counterpart, bracketing
//! the whole operation instead of refilling per region.

mod base;
mod fill;
mod pipeline;
mod resident;

pub use base::*;
// The function and the expansion module `#[cube]` derives beside it; one `use` covers both
// namespaces. The double-buffered walks fill their slots by hand, so they need it.
pub(crate) use fill::fill_operand;
pub use pipeline::*;
// fill is otherwise `Staging` fill/consume impls, resident adds `Tile::promote`; nothing else to
// re-export.
