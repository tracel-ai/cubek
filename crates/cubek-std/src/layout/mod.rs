mod addressing;
mod contiguous;
mod fragments;
mod row_major;

pub use addressing::{is_dense, is_nonoverlapping};
pub use contiguous::contiguous;
pub use fragments::*;
pub use row_major::RowMajorLayout;
