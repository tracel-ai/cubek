//! The verbs a client runs over tiles: [`matmul`] (`mma`), `mul`, [`softmax`] and `rows`.
//! Each reads an already-structured [`Tile`](crate::Tile) and either walks its levels or runs at
//! the leaf; the shared machinery they compose lives in [`crate::stage`].
//!
//! Moving cells is no verb of its own: [`Tile::copy_from`](crate::Tile::copy_from), which also
//! decodes a source that carries a table or scales.

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
