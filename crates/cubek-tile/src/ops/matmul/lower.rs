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
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final(_) => mma_leaf(self, lhs, rhs),
            Partitioner::Level(level) => {
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
