//! The leaf contraction `acc += lhs · rhs`, reached only at a *final* tile: the point where the
//! type-agnostic [lowering](super::lower) stops recursing and the tile finally has to admit it
//! holds numbers. Two peer leaves exist, picked by the accumulator's storage: a cmma fragment runs
//! the hardware instruction ([`cmma`](super::cmma)); plain `Gmem`/`Smem` runs a software
//! microkernel straight over the backing memory ([`register`](super::register)). Both are
//! terminal; neither tiles further.

use cubecl::prelude::*;

use crate::{matmul::instruction::register::mma_register_memory, *};

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
        // A partition that reaches a final tile carries exactly one fragment (a wider one
        // is consumed earlier, by the static walk at its partition level).
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
                 promote it first (Tile::promote / Resident::contract)"
            ));
            mma_register_memory::<E, EL, ER>(g, lhs, rhs, space)
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
    }
}
