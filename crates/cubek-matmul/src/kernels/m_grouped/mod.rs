//! The M-grouped product: rows sorted by group against the slab of a stacked rhs each group
//! names, launched as the dense product of the groups padded to chunks of rows, one chunk a
//! batch, the stack read through the chunks' table. The output can stay padded
//! ([`PaddedRows`](crate::kernels::grouped_rows::PaddedRows)) where the next step reads it so.

mod row_source;
mod settings;
mod tensors;
mod tune;

pub use row_source::*;
pub use settings::*;
pub use tensors::*;
pub use tune::*;
