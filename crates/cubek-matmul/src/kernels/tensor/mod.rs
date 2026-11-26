//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]), one of 8-bit floats under
//! scales a sum takes once ([`ScaledTensor`]), and any of them as it is stored ([`StoredTensor`]). Beside them, the matrices a producer of rows writes a product's lhs
//! into ([`LhsOutput`]).

mod lhs_rows;
mod packed;
mod plain;
mod scaled;
mod stored;

pub use lhs_rows::*;
pub use packed::*;
pub use plain::*;
pub use scaled::*;
pub use stored::*;
