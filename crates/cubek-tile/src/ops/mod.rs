//! The verbs a client runs over tiles: [`matmul`] (`mma`), [`mul`], [`softmax`] and [`rows`].
//! Each reads an already-structured [`Tile`](crate::Tile) and either walks its levels or runs at
//! the leaf; the shared machinery they compose lives in [`crate::load`].
//!
//! Moving cells is no verb of its own: a tile is filled from another through
//! [`Tile::copy_from`](crate::Tile::copy_from), which a quantized store dequantizes on the way.

mod attention;
pub(crate) mod matmul;
mod mul;
mod reduce;
mod rows;
mod softmax;
mod team;

pub use attention::*;
pub use matmul::*;
pub use reduce::*;
pub use softmax::*;
pub use team::*;
// mul and rows add `Tile` impls only; nothing to re-export.
