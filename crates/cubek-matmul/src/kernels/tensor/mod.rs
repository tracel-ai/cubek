//! The operands the routines read: a float one as cubecl binds it ([`PlainTensor`]), a packed
//! one bound as one tile of its words under its scales ([`PackedTensor`]), and either as it is
//! stored ([`StoredTensor`]).

mod packed;
mod plain;
mod stored;

pub use packed::*;
pub use plain::*;
pub use stored::*;
