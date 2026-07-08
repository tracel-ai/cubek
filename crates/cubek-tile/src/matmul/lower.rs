//! Lowering `c.mma(a, b)`: while the tile still has levels it lowers per its [`Schedule`],
//! shuffling operands around as opaque [`CubePrimitive`] tiles; at a final tile it hands off to the
//! [`mma_leaf`](super::instruction::mma_leaf), the one place that commits to concrete numeric types.

use cubecl::prelude::*;

use super::acc::{cmma_acc, cmma_drain};
use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::{matmul::instruction::mma_leaf, *};

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.mma(a, b)`: while levels remain, lower per the tile's [`Schedule`]; at a final tile,
    /// contract via the [`mma_leaf`] dispatch. A memory accumulator bound for the
    /// [`Leaf::Cmma`] instruction is hoisted into a resident fragment at this boundary
    /// ([`cmma_acc`]), walked in place of the tile, and drained back after.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final(_) => mma_leaf(self, lhs, rhs),
            Partitioner::Level(level) => {
                let is_mem = self.is_mem();
                let hoist = comptime!(is_mem && self.space.partitioner().leaf() == Leaf::Cmma);
                if hoist {
                    let mut acc = cmma_acc(self, lhs);
                    acc.mma(lhs, rhs);
                    cmma_drain(self, &acc);
                } else {
                    // The level's operation space is the merge of the operands' runtime spaces;
                    // the output contributes no axis beyond `lhs ∪ rhs`, so the two operands
                    // cover it.
                    let space = lhs.runtime_space().merge_with(&rhs.runtime_space());
                    match level.schedule() {
                        Schedule::Direct => mma_direct(lhs, rhs, self, space),
                        Schedule::Staged => mma_staged(lhs, rhs, self, space),
                        Schedule::DoubleBuffered => mma_double(lhs, rhs, self, space),
                    }
                }
            }
        }
    }
}
