//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`.
//!
//! Two layers: [`lower`] + [`schedule`] recurse over tiles as opaque [`CubePrimitive`], shuffling
//! operands until a final tile; [`leaf`] is the one place that commits to numbers, dispatching by
//! storage to the [`cmma`] (tensor-core) or [`register`] (software) backend.

mod instruction;
mod lower;
mod schedule;
