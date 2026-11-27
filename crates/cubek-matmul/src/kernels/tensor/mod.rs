//! The operands the routines read: one type, [`StoredTensor`], for any operand, its values stored
//! under its scheme (floats under no scale, 8-bit floats under scales a sum takes once, or packed
//! words under their scales), each buffer bound as [`PlainTensor`]. Beside it, the matrices a
//! producer of rows writes a product's lhs into ([`LhsOutput`]).

mod lhs_rows;
mod plain;
mod scaled;
mod stored;

pub use lhs_rows::*;
pub use plain::*;
pub use scaled::{PACKED_FP8_TILES_UNREAD, SCALES_WITHIN_CONTRACTION};
pub use stored::*;
