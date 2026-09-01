//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`.
//!
//! Two layers: [`lower`] + [`schedule`] recurse over tiles as opaque [`CubePrimitive`]
//! until a final tile, lowering to [`mma_leaf`](crate::mma_leaf).

mod distributed;
mod lower;
mod schedule;

pub(crate) use distributed::*;
