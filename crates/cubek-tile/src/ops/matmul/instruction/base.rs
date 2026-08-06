//! The leaf contraction `acc += lhs · rhs`, reached only at a *final* tile. Two peer
//! leaves, picked by the accumulator's storage: a cmma fragment runs the hardware
//! instruction ([`cmma`](super::cmma)); plain `Gmem`/`Smem` runs a software microkernel
//! ([`register`](super::register)).

use cubecl::prelude::*;

use super::register::mma_register_memory;
use crate::*;

/// What a plane leaf's 2-D single-`K` deduction rules out.
const PLANE_2D: &str = "mma: a plane (cmma/mma) leaf reads one contracted axis off a directly \
                        addressed operand; a gather-reduce runs on the register leaf";

/// The leaf contraction `acc += lhs · rhs`. Dispatch is dynamic on the accumulator's comptime
/// storage config
#[cube]
pub(crate) fn mma_leaf<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) {
    let space = comptime!(acc.space.clone());
    // A gather-reduce reaches the register leaf only. The plane leaves read a fragment grid whose
    // shape is deduced 2-D and single-contraction: `Space::contraction` asserts one contracted
    // axis, `PlanePartition::store` infers the A/B role from where that axis sits in the trailing
    // two, and `partition_grid` reads the grid off those same two. So both halves of the deduction
    // are refused here rather than left to fail silently: a wider reduce nest, and an operand
    // whose buffer a logical coordinate does not address directly. Generalizing the plane leaves
    // is a phase of its own.
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let plane_2d = comptime!(
        !lhs_gathered
            && !rhs_gathered
            && Space::contracted(&[&lhs.space, &rhs.space], &space).len() == 1
    );
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::PlaneTile(t) => {
            comptime!(assert!(plane_2d, "{}", PLANE_2D));
            t.mma(lhs, rhs)
        }
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(plane_2d, "{}", PLANE_2D));
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs)
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
    /// Contract this plane tile: the only place the two encodings' executes diverge.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(&mut self, lhs: &Tile<EL>, rhs: &Tile<ER>) {
        match self {
            PlaneTile::Cmma(d) => d.mma(lhs, rhs),
            PlaneTile::Mma(d) => d.mma(lhs, rhs),
            PlaneTile::Register(d) => d.mma(lhs, rhs),
        }
    }
}
