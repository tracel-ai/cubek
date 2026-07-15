//! The verbs a client runs over tiles: [`matmul`] (`mma`), [`softmax`], [`scalar`], [`dequantize`].
//! Each reads an already-structured [`Tile`](crate::Tile) and either walks its levels or runs at
//! the leaf; the shared machinery they compose lives in [`crate::staging`].

mod dequantize;
mod matmul;
mod scalar;
mod softmax;

pub use dequantize::*;
pub use scalar::*;
pub use softmax::*;
// matmul adds `Tile::mma` impls only — nothing to re-export.
