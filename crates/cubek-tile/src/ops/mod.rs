//! The verbs a client runs over tiles: [`matmul`] (`mma`), `mul`, [`softmax`] and `rows`.

pub mod matmul;
pub(crate) mod mul;
pub(crate) mod reduce;
pub(crate) mod row_factors;
pub(crate) mod rows;
pub mod softmax;
pub(crate) mod team;

pub(crate) use matmul::*;
pub(crate) use softmax::*;
// mul, rows and row_factors add `Tile` impls only; nothing to re-export.

pub use team::TeamUnit;
