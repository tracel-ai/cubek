//! How an operand lives across the contraction, i.e. what its [`Residence`](crate::Residence)
//! costs: a staging slot ([`base`]) sequenced by a [`pipeline`], built and driven by [`fill`]
//! (streamed vs fixed, materialized vs read where it lies), and scheduled as a depth-`n` software pipeline
//! by [`ring`]. [`AccumulatorScope`] is the output counterpart, reading a residence as the scope the
//! accumulator lives in instead of a refill per region.

mod accumulator;
mod base;
mod fill;
mod pipeline;
mod ring;

pub use accumulator::*;
pub use base::*;
pub use pipeline::*;
pub use ring::*;
// fill is `Ring`/`Staging` construction and fill/consume impls; nothing else to re-export.
