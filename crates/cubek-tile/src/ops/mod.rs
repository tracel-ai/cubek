//! The verbs a client runs over tiles: [`matmul`] (`mma`), [`mul`], [`softmax`] and [`copy`]. Each reads
//! an already-structured [`Tile`](crate::Tile) and either walks its levels or runs at the leaf;
//! the shared machinery they compose lives in [`crate::staging`]. Dequantization is not a verb: a
//! quantized store dequantizes under the plain [`Tile::copy`](crate::Tile::copy).

mod attention;
mod copy;
mod matmul;
mod mul;
mod normalize;
mod reduce;
mod softmax;

pub use attention::*;
pub use normalize::*;
pub use softmax::*;
// matmul, mul and reduce add `Tile` impls only; nothing to re-export.
