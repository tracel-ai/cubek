//! The attention verb's software leaves, one per fold shape:
//!
//! * [`score_columns`](crate::Tile::score_columns) and [`mix_columns`](crate::Tile::mix_columns):
//!   the shared-memory fold's two matmuls (the score into a materialized tile, the value mix
//!   with the online-softmax rescale fused in) under the software instruction, where a unit
//!   owns every `CUBE_DIM_X`-th column. Prefill's shape: real query blocks, a score tile between
//!   two matmuls.
//! * [`stream`]: the register-resident fold. No score tile and no barriers; each position's dot
//!   closes with a plane sum and feeds the accumulator immediately. Decode's shape: a single query
//!   position per group.
//!
//! The hardware form has no leaf of its own: the two matmuls are the general contraction
//! ([`cmma_accumulator`](crate::Tile::cmma_accumulator), [`mma`](crate::Tile::mma)) on a
//! plane-resident accumulator, and the rescale between them is
//! [`rescale_rows`](crate::Tile::rescale_rows) on that accumulator. Like
//! [`softmax`](crate::Tile::softmax) these are leaf ops: the caller owns the walk and the syncs.

mod columns;
mod stream;

pub use stream::*;
// columns adds `Tile` impls only; nothing to re-export.
