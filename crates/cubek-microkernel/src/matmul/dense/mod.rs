//! The one product, `lhs · rhs`, whichever of its inputs are packed: its microkernels, a folder per scope
//! ([`unit`](mod@unit), [`plane`], [`cube`], [`hypercube`]), each named by algorithm.

pub mod cube;
pub mod hypercube;
pub mod plane;
pub mod unit;
