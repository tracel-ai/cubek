//! `RhsTile` enum used by the partition-matmul body. The dispatching
//! per-partition allocators (`allocate_lhs_fragment` etc.) that match on
//! the matmul-kind enum live in `cubek-matmul` alongside the
//! `TileMatmul` enum they dispatch on; cubek-std only exposes the
//! per-variant primitives (e.g. `cmma_allocate_lhs`, `mma_allocate_lhs`).

use cubecl::prelude::*;

#[derive(CubeType)]
/// Per-partition rhs fragments. Single buffering keeps one fragment;
/// double buffering keeps two and rotates between them to overlap loads
/// with computes.
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
