//! Lowering `c.mma(a, b)`: at a final tile, the leaf instruction; while levels remain,
//! walk this level under its [`Schedule`]. Register residency is the kernel's explicit
//! bracket ([`promote`](Tile) … [`copy_from`](Tile::copy_from)), not a lowering decision.

use cubecl::prelude::*;

use super::instruction::mma_leaf;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.mma(a, b)`: contract at a final tile, else walk this level.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        self.mma_reduced(lhs, rhs, false)
    }

    /// [`mma`](Tile::mma) told whether the lanes hold disjoint K-slices of this tile, which only
    /// the level above can know (a leaf's partitioner is `Final`, its distributions consumed).
    /// The split is a leaf contract, so a level that still has to walk refuses to carry it.
    pub(crate) fn mma_reduced<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] split_k: bool,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final(_) => mma_leaf(self, lhs, rhs, split_k),
            Partitioner::Level(level) => {
                comptime!(assert!(
                    !split_k,
                    "mma: split-K must reach the leaf directly; this level still walks"
                ));
                // The level's operation space is the merge of the operands' runtime
                // spaces; the output contributes no axis beyond `lhs ∪ rhs`.
                let op_space = lhs.runtime_space().merge_with(&rhs.runtime_space());
                match comptime!(level.schedule()) {
                    Schedule::Direct => self.mma_direct(lhs, rhs, op_space),
                    Schedule::Staged => self.mma_staged(lhs, rhs, op_space),
                    Schedule::DoubleBuffered => self.mma_double(lhs, rhs, op_space),
                }
            }
        }
    }
}
