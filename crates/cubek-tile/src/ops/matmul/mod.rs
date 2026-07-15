//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`.
//!
//! Two layers: [`lower`] + [`schedule`] recurse over tiles as opaque [`CubePrimitive`]
//! until a final tile; [`instruction`] is the one place that commits to numbers,
//! dispatching by storage to the tensor-core or software-register backend.

mod instruction;
mod lower;
mod schedule;
