//! Lowering `c.mma(a, b)`: while the tile still has levels it lowers per its [`Schedule`],
//! shuffling operands around as opaque [`CubePrimitive`] tiles; at a final tile it hands off to the
//! [`Mma`](super::leaf::Mma) leaf, the one place that commits to concrete numeric types.

use cubecl::prelude::*;

use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::{matmul::leaf::Mma, *};

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
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => mma_direct(lhs, rhs, self),
                Schedule::Staged => mma_staged(lhs, rhs, self),
                Schedule::DoubleBuffered => mma_double(lhs, rhs, self),
            },
        }
    }

    /// The [`Direct`](Schedule::Direct) lowering's per-region step.
    pub fn mma_at<Lhs: CubePrimitive, Rhs: CubePrimitive>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        region: &Region,
    ) where
        Acc: Mma<Lhs, Rhs>,
    {
        self.at(region).mma(&lhs.at(region), &rhs.at(region));
    }
}
