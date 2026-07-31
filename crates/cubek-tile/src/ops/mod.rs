//! The verbs a client runs over tiles: [`matmul`] (`mma`) and [`softmax`]. Each reads an
//! already-structured [`Tile`](crate::Tile) and either walks its levels or runs at the leaf; the
//! shared machinery they compose lives in [`crate::staging`]. Dequantization is not a verb: a
//! quantized store dequantizes under the plain [`Tile::copy_from`](crate::Tile::copy_from).

mod attention;
mod matmul;
mod softmax;

pub use attention::*;
pub use softmax::*;
// matmul adds `Tile` impls only; nothing to re-export.
