//! Lowering `c.mma(a, b)`: while the tile still has levels it lowers per its [`Schedule`],
//! shuffling operands around as opaque [`CubePrimitive`] tiles; at a final tile it hands off to the
//! [`Mma`](super::instruction::Mma) leaf, the one place that commits to concrete numeric types.

use cubecl::prelude::*;

use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::{matmul::instruction::Mma, *};

#[cube]
impl<Acc: CubePrimitive> Tile<Acc> {
    /// `c.mma(a, b)`: while levels remain, lower per the tile's [`Schedule`]; at a final tile,
    /// contract via the [`Mma`] leaf.
    pub fn mma<Lhs: CubePrimitive, Rhs: CubePrimitive>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>)
    where
        Acc: Mma<Lhs, Rhs>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => Acc::mma(self, lhs, rhs),
            Partitioner::Level(level) => {
                // The level's operation space is the merge of the operands' runtime spaces; the
                // output contributes no axis beyond `lhs ∪ rhs`, so the two operands cover it.
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
