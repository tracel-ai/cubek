//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), and a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]).

mod packed;
mod plain;

pub use packed::*;
pub use plain::*;
