//! The M-grouped product: rows sorted by group against the slab of a stacked rhs each group
//! names, launched as the dense product of the groups padded to chunks of rows, one chunk a
//! batch, the stack read through the chunks' table. The output can stay padded
//! ([`PaddedRows`]) where the next step reads it so.

mod grouped_rows;
mod row_source;
mod settings;
mod tensors;
mod tune;

#[cfg(test)]
mod tests;

pub use grouped_rows::{
    COMBINED_ROUTES_UNREAD, GATED_ROWS_UNREAD, GROUPED_ROWS_UNREAD, GROUPED_TILE_OF_NO_ROW,
    GroupedRows, PaddedRows, ROUTE_SORT_UNREAD, RouteSort,
};
pub use row_source::*;
pub use settings::*;
pub use tensors::*;
pub use tune::*;

pub(crate) use grouped_rows::upload;
