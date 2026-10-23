//! Rows grouped by slab, laid out for a grouped product: every group padded to start on a whole
//! tile, and the slab of each tile. Nothing here is decided, so there is no routine: each kernel
//! is a flat pass over the padded rows.

mod kernel;
mod layout;
mod padded_rows;

#[cfg(test)]
mod tests;

pub use layout::{GroupedRows, RouteSort};
pub use padded_rows::PaddedRows;

pub(crate) use layout::upload;
