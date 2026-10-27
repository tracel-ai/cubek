mod addressing;
mod contiguous;
mod fragments;
mod row_major;
pub mod stored;
pub mod tile_order;

pub use addressing::{is_dense, is_nonoverlapping};
pub use contiguous::contiguous;
pub use fragments::*;
pub use row_major::RowMajorLayout;
