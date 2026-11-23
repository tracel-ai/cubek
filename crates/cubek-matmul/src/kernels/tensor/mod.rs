//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]), 8-bit float values
//! under one scale a row ([`RowScaledTensor`]), and any of them as it is stored
//! ([`StoredTensor`]). Beside them, the matrices a producer of rows writes a product's lhs
//! into ([`LhsOutput`]).

mod lhs_rows;
mod packed;
mod plain;
mod row_scaled;
mod stored;

pub use lhs_rows::*;
pub use packed::*;
pub use plain::*;
pub use row_scaled::*;
pub use stored::*;
