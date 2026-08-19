//! Generic axis reduction operations on [`Tile`](crate::Tile).
//!
//! Provides [`Tile::reduce_axis`](crate::Tile::reduce_axis), which reduces an input tile across
//! its contracted axes into an accumulator tile, walking spatial hierarchy levels down to the
//! register nest ([`microkernel::reduce`](crate::microkernel::reduce)) at
//! [`Partitioner::Final`](crate::Partitioner::Final).

mod lower;
mod schedule;
