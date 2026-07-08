//! Lowering `c.mma(a, b)`: each call classifies itself comptime ([`Tile::lowering`]) from the
//! accumulator's kind and its space, then takes exactly one step — the leaf instruction, the
//! resident-accumulator lifecycle, the partition microkernel, or a level walk under this
//! level's [`Schedule`].

use cubecl::prelude::*;

use super::instruction::{mma_leaf, mma_partition};
use super::resident::{mma_resident, partition_level};
use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::*;

/// The lowering step one `mma` call takes, decided comptime.
enum Lowering {
    /// A final tile: the instruction ([`mma_leaf`]).
    Leaf,
    /// A memory accumulator bound for [`Leaf::Cmma`]: run the whole contraction on a
    /// resident fragment partition ([`mma_resident`]).
    Resident,
    /// A fragment partition at its partition level: the comptime microkernel
    /// ([`mma_partition`] — fragments cannot be indexed by a runtime walk).
    Partition,
    /// Walk this level's regions under its [`Schedule`].
    Walk(Schedule),
}

impl Lowering {
    /// A memory accumulator: hoist to a resident partition when the leaf is cmma.
    fn mem(space: &Space) -> Lowering {
        match space.partitioner() {
            Partitioner::Final(_) => Lowering::Leaf,
            Partitioner::Level(level) => {
                if space.partitioner().leaf() == Leaf::Cmma {
                    Lowering::Resident
                } else {
                    Lowering::Walk(level.schedule())
                }
            }
        }
    }

    /// A fragment partition: the comptime microkernel at its partition level.
    fn partition(space: &Space) -> Lowering {
        match space.partitioner() {
            Partitioner::Final(_) => Lowering::Leaf,
            Partitioner::Level(level) => {
                if partition_level(space).is_some() {
                    Lowering::Partition
                } else {
                    Lowering::Walk(level.schedule())
                }
            }
        }
    }

    /// A plain fragment: leaf or walk.
    fn plain(space: &Space) -> Lowering {
        match space.partitioner() {
            Partitioner::Final(_) => Lowering::Leaf,
            Partitioner::Level(level) => Lowering::Walk(level.schedule()),
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The [`Lowering`] step this accumulator takes: each kind tells its own.
    fn lowering(&self) -> comptime_type!(Lowering) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => comptime!(Lowering::mem(&self.space)),
            TileKind::CmmaPartition(_) => comptime!(Lowering::partition(&self.space)),
            TileKind::Cmma(_) => comptime!(Lowering::plain(&self.space)),
            TileKind::TmaGmem(_) => comptime!(panic!("mma: a tma source is not an accumulator")),
        }
    }

    /// `c.mma(a, b)`: take this call's [`Lowering`] step.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let lowering = self.lowering();
        match comptime!(lowering) {
            Lowering::Leaf => mma_leaf(self, lhs, rhs),
            Lowering::Resident => mma_resident(self, lhs, rhs),
            Lowering::Partition => mma_partition(self, lhs, rhs),
            Lowering::Walk(schedule) => {
                // The level's operation space is the merge of the operands' runtime spaces;
                // the output contributes no axis beyond `lhs ∪ rhs`, so the two operands
                // cover it.
                let space = lhs.runtime_space().merge_with(&rhs.runtime_space());
                match schedule {
                    Schedule::Direct => mma_direct(lhs, rhs, self, space),
                    Schedule::Staged => mma_staged(lhs, rhs, self, space),
                    Schedule::DoubleBuffered => mma_double(lhs, rhs, self, space),
                }
            }
        }
    }
}
