//! The attention verb's leaves, one per fold shape:
//!
//! * [`score_columns`](crate::Tile::score_columns) / [`score_fragments`](crate::Tile::score_fragments)
//!   and [`mix_columns`](crate::Tile::mix_columns) / [`mix_fragments`](crate::Tile::mix_fragments):
//!   the shared-memory fold's two matmuls (the score into a materialized tile, the value mix
//!   with the online-softmax rescale fused in), each in its software and hardware form.
//!   Prefill's shape: real query blocks, a score tile between two matmuls.
//! * [`stream`]: the register-resident fold. No score tile and no barriers; each position's dot
//!   closes with a plane sum and feeds the accumulator immediately. Decode's shape: a single query
//!   position per group.
//!
//! The kernel picks each matmul's form: the software instruction runs [`columns`], where a unit
//! owns every `CUBE_DIM_X`-th column, and a hardware one runs [`fragments`], where a plane owns
//! every `planes`-th fragment. Like [`softmax`](crate::Tile::softmax) these are leaf ops on
//! shared-memory tiles: the caller owns the walk and the syncs.

mod columns;
mod fragments;
mod stream;

pub use fragments::{FragmentOwnership, FragmentShape};
pub use stream::*;
// columns and fragments add `Tile` impls only; nothing to re-export.
