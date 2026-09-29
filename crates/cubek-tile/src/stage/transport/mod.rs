//! Moving an operand's cells into the tile an instruction reads them from: which transport a
//! pairing takes ([`base`]), the four that do the work, the straight one split around a
//! contraction ([`prefetch`]), and the copy engine a straight one can hand its lines to
//! ([`async_copy`]).

pub(crate) mod async_copy;
pub(crate) mod base;
pub(crate) mod cooperative;
pub(crate) mod padded;
pub(crate) mod prefetch;
pub(crate) mod scaled;
pub(crate) mod scanned;

pub(crate) use prefetch::{MOST_FETCHED_SCALARS, UnitLines};
