//! How an operand lives across the contraction, i.e. what its [`Residence`](crate::Residence)
//! costs: a staging slot ([`base`]) sequenced by a [`pipeline`], built and driven by [`fill`]
//! (streamed vs fixed, materialized vs read where it lies), and scheduled as a depth-`n` software pipeline
//! by [`ring`]. [`resident`] (`promote`) is the output counterpart, bracketing the whole operation
//! instead of refilling per region.

mod base;
mod fill;
mod pipeline;
mod resident;
mod ring;

pub use base::*;
pub use pipeline::*;
pub use ring::*;
// fill is `Ring`/`Staging` construction and fill/consume impls, resident adds `Tile::promote`;
// nothing else to re-export.
