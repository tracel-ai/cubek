//! The leaf contraction `acc += lhs · rhs`, reached only at a *final* tile. Two peer
//! leaves, picked by the accumulator's storage: a cmma fragment runs the hardware
//! instruction ([`cmma`](super::cmma)); plain `Gmem`/`Smem` runs a software microkernel
//! ([`register`](super::register)).

use cubecl::prelude::*;

use super::register::mma_register_memory;
use crate::*;

/// The leaf contraction `acc += lhs · rhs`. Dispatch is dynamic on the accumulator's comptime
/// storage config
#[cube]
pub(crate) fn mma_leaf<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::Cmma(d) => d.mma(lhs, rhs),
        // A partition that reaches a final tile carries exactly one fragment; a wider
        // one is consumed earlier, at its partition level.
        TileKind::CmmaPartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            p.at(0usize, 0usize).mma(lhs, rhs)
        }
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            comptime!(assert!(
                space.partitioner().leaf() == Leaf::Register,
                "mma: a cmma-leaf accumulator runs register-resident — \
                 promote it first (Tile::promote), copy it back after"
            ));
            mma_register_memory::<E, EL, ER>(g, lhs, rhs, space, false)
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
    }
}

/// The leaf contraction with an intra-plane K-reduction: the K axis was `Cut::unit`, so each
/// lane holds a disjoint K-slice contributing to the same `(m, n)` cell. The register microkernel
/// seeds from zero, `plane_sum`-reduces across lanes, and lets lane 0 write. Register-resident
/// only (a cmma fragment has no such split); [`mma_direct`](crate::Tile::mma_direct) reaches this
/// only when the split is comptime-real (resolved `plane_size > 1`).
#[cube]
pub(crate) fn mma_leaf_split<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            comptime!(assert!(
                space.partitioner().leaf() == Leaf::Register,
                "mma: split-K is register-resident only"
            ));
            mma_register_memory::<E, EL, ER>(g, lhs, rhs, space, true)
        }
        _ => panic!("mma: split-K (Cut::unit on K) is supported only by the register leaf"),
    }
}
