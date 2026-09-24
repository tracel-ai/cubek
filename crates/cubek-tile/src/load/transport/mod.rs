//! Moving an operand's cells into the tile an instruction reads them from: which transport a
//! pairing takes ([`base`]), and the four that do the work.

pub(crate) mod base;
pub(crate) mod cooperative;
pub(crate) mod padded;
pub(crate) mod scanned;
pub(crate) mod words;
