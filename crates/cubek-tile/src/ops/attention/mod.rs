//! The attention verb's leaves, one per fold shape:
//!
//! * [`score`](crate::Tile::score) and [`mix`](crate::Tile::mix): the shared-memory fold's two
//!   matmuls (the score into a materialized tile, the value mix with the online-softmax rescale
//!   fused in). Prefill's shape: real query blocks, a score tile between two matmuls.
//! * [`stream`]: the register-resident fold. No score tile and no barriers; each position's dot
//!   closes with a plane sum and feeds the accumulator immediately. Decode's shape: a single query
//!   position per group.
//!
//! Both matmuls contract through the instruction their accumulator's space states, and nothing
//! else picks it: the software instruction runs [`columns`], where a unit owns every
//! `CUBE_DIM_X`-th column, and a hardware one runs [`fragments`], where a plane owns every
//! `planes`-th fragment. Like [`softmax`](crate::Tile::softmax) these are leaf ops on
//! shared-memory tiles: the caller owns the walk and the syncs.

mod base;
mod columns;
mod fragments;
mod stream;

pub use stream::*;
// base, columns and fragments add `Tile` impls only; nothing to re-export.
