//! Generic axis reduction operations on [`Tile`](crate::Tile).
//!
//! Provides [`Tile::reduce_axis`](crate::Tile::reduce_axis), which reduces an input tile across
//! its contracted axes into an accumulator tile at a final tile, through the register nest
//! ([`instruction::registers::reduce`](crate::instruction::registers::reduce)).

mod lower;
