//! Rows grouped by slab, laid out for a grouped product: every group padded to start on a whole
//! tile, and the slab of each tile. Nothing here is decided, so there is no routine: each kernel
//! is a flat pass over the padded rows.

mod kernel;
mod layout;
mod padded_rows;

#[cfg(test)]
mod tests;

pub use layout::{
    COMBINED_ROUTES_UNREAD, GATED_ROWS_UNREAD, GROUPED_ROWS_UNREAD, GROUPED_TILE_OF_NO_ROW,
    GroupedRows, ROUTE_SORT_UNREAD, RouteSort,
};
pub use padded_rows::PaddedRows;

pub(crate) use layout::upload;
