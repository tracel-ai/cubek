//! Lowering `c.mma(a, b)`: at a final tile, the leaf instruction; while levels remain, a
//! memory accumulator bound for [`Leaf::Cmma`] first moves into registers ([`Resident`]),
//! then every level is walked under its [`Schedule`] — each schedule picks the tier its
//! accumulator lives in (see [`schedule`](super::schedule)).

use cubecl::prelude::*;

use super::instruction::mma_leaf;
use super::schedule::{contracted_extent, mma_direct, mma_double, mma_staged};
use crate::*;

/// Enter register residency for `out` and run its contraction there: the recursion
/// continues on the register tile in place of the memory one (it carries the same space,
/// so the schedules walk it exactly like the tile it replaces), and the write-back
/// follows inside [`contract`](Resident::contract).
#[cube]
fn mma_resident<Acc: Numeric, Lhs: Numeric, Rhs: Numeric>(
    out: &mut Tile<Acc>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    let k = comptime!(contracted_extent(&lhs.space, &out.space));
    let mut acc = out.promote(k);
    acc.contract(out, |frags| frags.mma(lhs, rhs));
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Whether this accumulator must first move into registers: memory bound for the cmma
    /// leaf runs the whole contraction on a resident accumulator. Comptime.
    fn enters_residency(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => {
                comptime!(self.space.partitioner().leaf() == Leaf::Cmma)
            }
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => comptime!(false),
            TileKind::TmaGmem(_) => comptime!(panic!("mma: a tma source is not an accumulator")),
        }
    }

    /// `c.mma(a, b)`: contract at a final tile, else walk this level.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final(_) => mma_leaf(self, lhs, rhs),
            Partitioner::Level(level) => {
                let resident = self.enters_residency();
                if resident {
                    mma_resident(self, lhs, rhs);
                } else {
                    // The level's operation space is the merge of the operands' runtime
                    // spaces; the output contributes no axis beyond `lhs ∪ rhs`, so the
                    // two operands cover it.
                    let space = lhs.runtime_space().merge_with(&rhs.runtime_space());
                    match comptime!(level.schedule()) {
                        Schedule::Direct => mma_direct(lhs, rhs, self, space),
                        Schedule::Staged => mma_staged(lhs, rhs, self, space),
                        Schedule::DoubleBuffered => mma_double(lhs, rhs, self, space),
                    }
                }
            }
        }
    }
}
