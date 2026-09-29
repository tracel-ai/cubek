//! The verbs a client runs over tiles: [`matmul`] (`mma`), `mul`, [`softmax`] and `rows`.

pub mod matmul;
mod mul;
pub mod reduce;
mod rows;
pub mod softmax;
mod team;

pub(crate) use matmul::*;
pub(crate) use softmax::*;
pub use team::*;
// mul and rows add `Tile` impls only; nothing to re-export.
