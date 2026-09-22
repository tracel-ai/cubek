//! Staging an operand across a walk: a staging slot ([`base`]) sequenced by a [`pipeline`],
//! built and filled by [`fill`] (streamed vs fixed), and scheduled as a depth-`n` software
//! pipeline by [`ring`].

mod base;
mod fill;
#[cfg(test)]
mod model;
mod pipeline;
mod ring;
mod storage;

pub use base::*;
pub use pipeline::*;
pub use ring::*;
pub use storage::*;
// fill is `Ring`/`Staging` construction and fill/consume impls; nothing else to re-export.
