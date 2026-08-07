//! The leaf contraction `acc += lhs · rhs`, reached only at a *final* tile. Two peer
//! leaves, picked by the accumulator's storage: a cmma fragment runs the hardware
//! instruction ([`cmma`](super::cmma)); plain `Gmem`/`Smem` runs a software microkernel
//! ([`register`](super::register)).

use cubecl::prelude::*;

use super::register::mma_register_memory;
use crate::*;

/// What a strided-window fragment's 2-D single-`K` deduction rules out.
const STRIDED_2D: &str = "mma: a cmma or plane-register fragment reads one contracted axis off a \
                          directly addressed operand; a gather or a wider reduce needs the \
                          manual-mma leaf, or an unpromoted Gmem/Smem accumulator, whose software \
                          microkernel is the `mma_register_memory` arm below";

/// The leaf contraction `acc += lhs · rhs`. Dispatch is dynamic on the accumulator's comptime
/// storage config
#[cube]
pub(crate) fn mma_leaf<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) {
    let space = comptime!(acc.space.clone());
    // Both operands flatten their contracted axes into one `k` edge by extent alone
    // (`matrix_split`), so listing the same axes in different orders would contract mismatched
    // positions with matching shapes. Vacuous for a single contracted axis.
    comptime!(assert!(
        Space::contraction_agrees(&lhs.space, &rhs.space, &space),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.space.contracting(&space),
        rhs.space.contracting(&space)
    ));
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space),
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space)
        }
        // A memory accumulator runs the software microkernel. A plane-form accumulator that was
        // never promoted lands in the arms above and meets their kind-pairing panics; there is no
        // second declaration left to check this one against.
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            mma_register_memory::<E, EL, ER>(g, lhs, rhs, space)
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile: the only place the encodings' executes diverge, and so the only
    /// place that knows which of them reads a raw strided window.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                strided_2d(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
            // Reads its operands element by element through `Tile::fragment_matrix`, so a gather
            // folds underneath and a wider reduce flattens into the `k` edge.
            PlaneTile::Mma(d) => d.mma(lhs, rhs),
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
        }
    }
}

/// Refuse what a raw strided window (`window_slice` + `row_stride`) cannot address: a gathered
/// operand, and a contraction over more than one axis.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered
            && !rhs_gathered
            && Space::contracted(&[&lhs.space, &rhs.space], &out).len() == 1,
        "{}",
        STRIDED_2D
    ));
}
