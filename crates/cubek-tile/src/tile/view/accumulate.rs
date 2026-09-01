use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::{instruction::plane, *};

/// What an accumulation starts from: the statement a verb makes about the cells it is about to
/// write ([`Tile::mm`] and [`Tile::reduce_axis`] say [`Identity`](InitFrom::Identity), the
/// accumulating verbs say [`Cell`](InitFrom::Cell)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum InitFrom {
    /// Fold onto the cell: it holds a value that counts, whether a partial the walk above left
    /// there or an accumulator the caller seeded.
    Cell,
    /// Start from the monoid's identity: nothing before this accumulation counts, so the cell is
    /// never read and the result is written outright.
    Identity,
}

/// Where the cell's own value enters the result. Exactly one site reads it, or neither does, and
/// both sites read this rather than deciding for themselves.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CellRead {
    /// This lane owns the cell whole, so the block starts from it.
    AtSeed,
    /// The plane's lanes each hold a partial, so no lane may start from the cell: the one lane
    /// elected to write folds it in once instead.
    AtCommit,
    /// Nothing the cell holds counts.
    Never,
}

impl CellRead {
    /// Derived, never stated: whether the cell counts at all is the accumulation's statement, and
    /// which site reads it is what the plane's lanes hold of it.
    ///
    /// A destination that folds is never read here, whatever the accumulation says. Folding *is*
    /// the read-modify-write, done atomically by the store, so reading the cell back to add to it
    /// would both duplicate what the commit does and race every other instance writing it. That
    /// is what lets an accumulation contract in place across a split: the cell may already hold
    /// other instances' contributions, and nothing here has to know.
    const fn of(lane_share: LaneShare, init_from: InitFrom, write: Write) -> Self {
        match write {
            Write::Accumulate => CellRead::Never,
            Write::Replace => match init_from {
                InitFrom::Identity => CellRead::Never,
                InitFrom::Cell => match lane_share {
                    LaneShare::Whole => CellRead::AtSeed,
                    LaneShare::Plane | LaneShare::Group { .. } => CellRead::AtCommit,
                },
            },
        }
    }
}

/// Which of the plane's lanes carry a drain's writes, and what they do to their values first.
///
/// Derived from the three facts that decide it and matched on once, so a write reads as four
/// cases rather than as a lane guard nested in a share. Shared by the two sites that carry an
/// accumulation into memory: this view's commit, and a register block's drain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Drain {
    /// Each lane holds whole cells of its own and writes them as they are.
    EachLane,
    /// Every lane holds the same whole cells, and the write folds, so one lane writes.
    LaneZero,
    /// The plane's lanes each hold a partial of one cell: combine across the plane, lane zero
    /// writes.
    PlaneFold,
    /// Groups of lanes each hold a partial of one cell: combine within the group, its first
    /// lane writes.
    GroupFold { fold_mask: usize },
}

impl Drain {
    pub(crate) const fn of(lanes: Lanes, write: Write) -> Self {
        match lanes.share {
            LaneShare::Plane => Drain::PlaneFold,
            LaneShare::Group { fold_mask } => Drain::GroupFold { fold_mask },
            // Nothing is folded across the lanes, so nothing has to be combined. Whether they may
            // all write is a different question, and the one a fold turns on: repeated lanes hold
            // the same cells, so a store lands the same value however many make it and a fold
            // lands it once per lane.
            LaneShare::Whole => match (lanes.work, write) {
                (LaneWork::Repeated, Write::Accumulate) => Drain::LaneZero,
                (LaneWork::Repeated, Write::Replace) | (LaneWork::Own, _) => Drain::EachLane,
            },
        }
    }
}

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. The write-side mirror of a
/// quantized view dequantizing on read: it owns the [`LaneShare`], so cells the plane's lanes
/// hold partials of combine on commit and the contraction never asks.
///
/// It owns the [`Monoid`] and the [`CellRead`] for the same reason. Both are one fact about the
/// accumulation, not a fact about each cell, so they are settled where the view is built and read
/// from there by every seed and commit the leaf runs.
#[derive(CubeType)]
pub(crate) struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedViewMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    lanes: Lanes,
    #[cube(comptime)]
    monoid: Monoid,
    #[cube(comptime)]
    cell_read: CellRead,
    #[cube(comptime)]
    drain: Drain,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedViewMut<'a, Vector<E, V>, C>,
        #[comptime] lanes: Lanes,
        #[comptime] split_share: SplitShare,
        #[comptime] write: Write,
        #[comptime] monoid: Monoid,
        #[comptime] init_from: InitFrom,
    ) -> Self {
        comptime!(split_share.validate(write, "AccumulateView"));
        AccumulateView::<'a, E, V, C> {
            values,
            lanes,
            monoid,
            cell_read: comptime!(CellRead::of(lanes.share, init_from, write)),
            drain: comptime!(Drain::of(lanes, write)),
        }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`].
    pub fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// How these cells are shared across the plane's lanes. A leaf that commits *conditionally*
    /// has to ask: past `Whole`, [`commit`](Self::commit) folds across the plane, and a plane op
    /// under divergent control flow is undefined.
    pub(crate) fn lane_share(&self) -> comptime_type!(LaneShare) {
        comptime!(self.lanes.share)
    }

    /// The monoid these cells fold under, stated where the view was built. A register block asks
    /// so its own seed and commit start from and collapse under the same fold this does.
    pub fn monoid(&self) -> comptime_type!(Monoid) {
        comptime!(self.monoid)
    }

    /// Whether a non-empty output block is wholly valid for unchecked seed/commit accesses.
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        self.values.block_in_bounds(pos, extent)
    }

    /// A block's starting value: the cell where this is the site that reads it, the monoid's
    /// identity everywhere else.
    pub fn seed(&self, pos: C) -> Vector<E, V> {
        match comptime!(self.cell_read) {
            CellRead::AtSeed => self.values.read(pos),
            CellRead::AtCommit | CellRead::Never => {
                Vector::<E, V>::cast_from(Monoid::identity::<E>(self.monoid))
            }
        }
    }

    /// Fold a finished block back. The fold reduces each `V`-wide cell element-wise
    /// and leaves every lane holding a partial of it with the total, so one of them writes and its
    /// siblings don't all hit the address: the plane's first lane where the whole plane shares one
    /// cell, each group's first lane where the plane carries a cell per group.
    pub fn commit(&mut self, pos: C, value: Vector<E, V>) {
        match comptime!(self.drain) {
            Drain::PlaneFold => {
                let combined = plane::broadcast::<Vector<E, V>>(value, self.monoid);
                self.commit_shared(pos, combined, UNIT_POS_X == 0);
            }
            Drain::GroupFold { fold_mask } => {
                let combined = plane::group(value, fold_mask, self.monoid);
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                self.commit_shared(pos, combined, lane_in_group == 0);
            }
            // Nothing to combine, but a fold from lanes that repeat each other's work would land
            // once per lane, so one of them makes it.
            Drain::LaneZero => self.commit_shared(pos, value, UNIT_POS_X == 0),
            Drain::EachLane => self.values.write(pos, value),
        }
    }

    /// Commit a cell the plane's lanes share, under the one lane elected to write it. Where this
    /// is the site that reads the cell, that lane folds it in, which no lane's seed could do.
    fn commit_shared(&mut self, pos: C, combined: Vector<E, V>, leader: bool) {
        if leader {
            match comptime!(self.cell_read) {
                CellRead::AtCommit => {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, self.monoid.fold::<Vector<E, V>>(old, combined));
                }
                CellRead::AtSeed | CellRead::Never => self.values.write(pos, combined),
            }
        }
    }
}
