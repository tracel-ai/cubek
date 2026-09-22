//! The attention verb's software leaves, one per fold shape:
//!
//! * [`score_columns`](crate::Tile::score_columns) and [`mix_columns`](crate::Tile::mix_columns):
//!   the shared-memory fold's two matmuls (score into a materialized tile, value mix with the
//!   softmax rescale fused), each unit owning every `CUBE_DIM_X`-th column. Prefill's shape.
//!
//! * [`stream`]: the register-resident fold. No score tile and no barriers; each position's dot
//!   closes with a plane sum and feeds the accumulator immediately. Decode's shape: one query
//!   position per group.
//!
//! [`splits`] closes a split walk of either shape: the teams publish their running states and
//! [`merge_splits`](crate::Tile::merge_splits) turns them into the cross-split weights the drain
//! contracts.
//!
//! The hardware form has no leaf of its own: the two matmuls are the general contraction
//! ([`cmma_accumulator`](crate::Tile::cmma_accumulator), [`mma`](crate::Tile::mma)) on a
//! plane-resident accumulator, with [`rescale_rows`](crate::Tile::rescale_rows) between them.
//!
//! Like [`softmax`](crate::Tile::softmax) these are leaf ops: the caller owns walk and syncs.

mod columns;
mod splits;
mod stream;

pub use stream::*;
// columns and splits add `Tile` impls only; nothing to re-export.
