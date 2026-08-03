//! The attention verb's leaves, one family per fold shape:
//!
//! * [`columns`]: the shared-memory fold's matmuls (score into a
//!   materialized tile, value mix with the rescale fused), at column
//!   ownership. Prefill's shape: real query blocks, a score tile between
//!   two matmuls.
//! * [`stream`]: the register-resident fold. No score tile and no barriers;
//!   each position's dot closes with a plane sum and feeds the accumulator
//!   immediately. Decode's shape: a single query position per group.

mod columns;
mod stream;

pub use stream::*;
// columns adds `Tile` impls only; nothing to re-export.
