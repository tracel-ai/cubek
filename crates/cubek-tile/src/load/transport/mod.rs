//! Moving an operand's cells into the tile an instruction reads them from: which transport a
//! pairing takes ([`base`]), the four that do the work, and the straight one split around a
//! contraction ([`prefetch`]).

pub(crate) mod base;
pub(crate) mod cooperative;
pub(crate) mod padded;
pub(crate) mod prefetch;
pub(crate) mod scanned;
pub(crate) mod words;

pub use prefetch::{MOST_FETCHED_SCALARS, UnitLines};
