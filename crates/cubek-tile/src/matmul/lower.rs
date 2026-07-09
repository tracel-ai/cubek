//! Lowering `c.mma(a, b)`: at a final tile, the leaf instruction; while levels remain, the
//! accumulator's backing store picks the level's step ([`LevelStep`]) — enter register
//! residency, the register-tier fragment walk, or an ordinary region walk under the
//! level's [`Schedule`].

use cubecl::prelude::*;

use super::instruction::mma_leaf;
use super::resident::{mma_resident, partition_level};
use super::schedule::{mma_direct, mma_double, mma_fragment_walk, mma_staged};
use crate::*;

/// How one level is taken, decided comptime by the accumulator's backing store. The
/// space alone can't tell: two of the steps exist only because fragments do (a memory
/// accumulator must move into them; once in them, only a comptime walk can index them).
enum LevelStep {
    /// A memory accumulator bound for [`Leaf::Cmma`]: promote it to a resident fragment
    /// partition for the whole contraction ([`mma_resident`]), then recurse.
    Resident,
    /// A fragment partition at its partition level: the comptime register-tier walk
    /// ([`mma_fragment_walk`] — fragments cannot be indexed by a runtime walk).
    FragmentWalk,
    /// Anything else: walk this level's regions under its [`Schedule`].
    Walk,
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The [`LevelStep`] this accumulator takes: each kind tells its own.
    fn level_step(&self) -> comptime_type!(LevelStep) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => {
                comptime!(if self.space.partitioner().leaf() == Leaf::Cmma {
                    LevelStep::Resident
                } else {
                    LevelStep::Walk
                })
            }
            TileKind::CmmaPartition(_) => {
                comptime!(if partition_level(&self.space).is_some() {
                    LevelStep::FragmentWalk
                } else {
                    LevelStep::Walk
                })
            }
            TileKind::Cmma(_) => comptime!(LevelStep::Walk),
            TileKind::TmaGmem(_) => comptime!(panic!("mma: a tma source is not an accumulator")),
        }
    }

    /// `c.mma(a, b)`: contract at a final tile, else take this level's [`LevelStep`].
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final(_) => mma_leaf(self, lhs, rhs),
            Partitioner::Level(level) => {
                let step = self.level_step();
                match comptime!(step) {
                    LevelStep::Resident => mma_resident(self, lhs, rhs),
                    LevelStep::FragmentWalk => mma_fragment_walk(lhs, rhs, self),
                    LevelStep::Walk => {
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
}
