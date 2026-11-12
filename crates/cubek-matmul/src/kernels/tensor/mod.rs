//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]), a float one multiplied by a
//! scale tensor as it is read ([`ScaledTensor`]), and each as it is
//! stored ([`StoredTensor`]). Beside them, the matrices a producer of rows writes a product's lhs
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
