//! The verbs a client runs over tiles: [`matmul`] (`mma`), `mul`, [`softmax`] and `rows`.

pub mod matmul;
mod mul;
pub mod reduce;
mod row_factors;
mod rows;
pub mod softmax;
mod team;

pub(crate) use matmul::*;
pub(crate) use softmax::*;
pub use team::*;
// mul, rows and row_factors add `Tile` impls only; nothing to re-export.
