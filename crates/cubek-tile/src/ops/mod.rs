//! The verbs a client runs over tiles: [`matmul`] (`mma`), [`mul`], [`softmax`], [`rows`] and
//! [`copy`]. Each reads an already-structured [`Tile`](crate::Tile) and either walks its levels or
//! runs at the leaf; the shared machinery they compose lives in [`crate::staging`].
//!
//! Dequantization is no verb: a quantized store dequantizes in [`Tile::copy`](crate::Tile::copy).

mod attention;
mod copy;
mod matmul;
mod mul;
mod normalize;
mod reduce;
mod rows;
mod softmax;

pub use attention::*;
pub use normalize::*;
pub use softmax::*;
// mul, reduce and rows add `Tile` impls only; nothing to re-export.
