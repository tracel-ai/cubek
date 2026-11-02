//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]), and either as it is
//! stored ([`StoredTensor`]). Beside them, the matrices a producer of rows writes a product's lhs
//! into ([`LhsOutput`]).

mod lhs_rows;
mod packed;
mod plain;
mod stored;

pub use lhs_rows::*;
pub use packed::*;
pub use plain::*;
pub use stored::*;
