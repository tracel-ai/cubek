//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`, at a final tile
//! ([`mma_leaf`](crate::mma_leaf)). The levels above are the kernel's own walk.

mod lower;
