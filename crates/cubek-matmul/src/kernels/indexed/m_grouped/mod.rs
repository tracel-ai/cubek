//! The M-grouped product: rows sorted by group against the slab of a stacked rhs each group
//! names, launched as the dense product of the groups padded to chunks of rows, one chunk a
//! batch, the stack read through the chunks' table. The output can stay padded
//! ([`PaddedRows`]) where the next step reads it so. A caller that measures it races the
//! candidates listed for it ([`MGroupedMatmulCandidates`]), the dense plane shuffle, which holds
//! a chunk of a group's rows against one read of its slab, against the cube staged routines,
//! which stage them, under a key of its own ([`MGroupedMatmulKey`]).

mod candidate;
mod grouped_rows;
mod key;
mod row_source;
mod settings;
mod tensors;

#[cfg(test)]
mod tests;

pub use candidate::*;
pub use grouped_rows::{GroupedRows, PaddedRows, RouteSort};
pub use key::*;
pub use row_source::*;
pub use settings::*;
pub use tensors::*;

pub(crate) use grouped_rows::upload;
