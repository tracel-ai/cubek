//! The attention verb's leaves. Two families, one per fold shape:
//!
//! * [`columns`] — the shared-memory fold's matmul leaves (score into a
//!   materialized tile, value mix with the rescale fused), at team-local
//!   column ownership. The shape prefill wants: real query blocks, a score
//!   tile between two matmuls.
//! * [`stream`] — the register-resident decode fold: no score tile, no
//!   barriers; each position's dot closes with a plane sum and the
//!   probability multiplies the value row immediately. The shape a single
//!   query position wants.

mod columns;
mod stream;

pub use stream::*;
// columns adds `Tile` impls only; nothing to re-export.
