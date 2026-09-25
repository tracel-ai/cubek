use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::*;

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
impl CubeDebug for InitFrom {}

/// Where the cell's own value enters the result. Exactly one site reads it, or neither does, and
/// both sites read this rather than deciding for themselves.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CellRead {
    /// This unit owns the cell whole, so the block starts from it.
    AtSeed,
    /// The plane's units each hold a partial, so no unit may start from the cell: the one unit
    /// elected to write folds it in once instead.
    AtCommit,
    /// Nothing the cell holds counts.
    Never,
}

impl CellRead {
    /// Derived, never stated: whether the cell counts at all is the accumulation's statement, and
    /// which site reads it is what the plane's units hold of it.
    ///
    /// A destination that folds is never read here, whatever the accumulation says: the store's
    /// atomic read-modify-write *is* the fold, so reading the cell back would duplicate the commit
    /// and race every other instance writing it. That is what lets a split contract in place.
    const fn of(unit_share: UnitShare, init_from: InitFrom, write: Write) -> Self {
        match write {
            Write::Accumulate => CellRead::Never,
            Write::Replace => match init_from {
                InitFrom::Identity => CellRead::Never,
                InitFrom::Cell => match unit_share {
                    UnitShare::Repeated | UnitShare::Whole => CellRead::AtSeed,
                    UnitShare::Plane | UnitShare::Group { .. } => CellRead::AtCommit,
                },
            },
        }
    }
}

/// How a plane-resident accumulator's tiles reach memory: stored straight through the intrinsic,
/// or bounced through the scratch, together where every tile has a slot of its own and one at a
/// time where they share one. Read off the [`Scratch`] the accumulator was opened with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPlan {
    /// No scratch: each tile stores through its own intrinsic.
    Straight,
    /// Every tile has its own slot, so every spill happens before any add: two barriers in all.
    BounceTogether,
    /// One slot between the tiles, so each spills and adds inside the loop: three barriers each.
    BounceEach,
}

impl DrainPlan {
    pub(crate) fn new(scratch: Scratch) -> Self {
        match scratch {
            Scratch::None => DrainPlan::Straight,
            Scratch::OneTile => DrainPlan::BounceEach,
            Scratch::WholeGrid => DrainPlan::BounceTogether,
        }
    }
}

/// What a drain does to one tile of the accumulator at the leaf of its descent: the two halves of
/// a bounce run as passes of their own where every tile has a slot, so the barriers between them
/// are the whole drain's rather than each tile's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPass {
    /// Store through the tile's own intrinsic.
    Copy,
    /// Spill, add, and the three barriers around them.
    Bounce,
    /// Spill alone: the first pass of a drain whose tiles each have a slot.
    Spill,
    /// Add alone: its second pass.
    Add,
}

/// Which of the plane's units carry a drain's writes, and what they do to their values first.
///
/// Derived from the three facts that decide it and matched on once, so a write reads as four
/// cases rather than as a unit guard nested in a share. Shared by the two sites that carry an
/// accumulation into memory: this view's commit, and a register block's drain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Drain {
    /// Each unit holds whole cells of its own and writes them as they are.
    EachUnit,
    /// Every unit holds the same whole cells, and the write folds, so one unit writes.
    UnitZero,
    /// The plane's units each hold a partial of one cell: combine across the plane, unit zero
    /// writes.
    PlaneFold,
    /// Groups of units each hold a partial of one cell: combine within the group, its first
    /// unit writes.
    GroupFold { fold_mask: usize },
}

impl Drain {
    pub(crate) const fn of(units: UnitShare, write: Write) -> Self {
        match (units, write) {
            (UnitShare::Plane, _) => Drain::PlaneFold,
            (UnitShare::Group { fold_mask }, _) => Drain::GroupFold { fold_mask },
            // Nothing is folded across the units, so nothing has to be combined. Whether they may
            // all write is what a fold turns on: repeated units hold the same cells, so a store
            // lands the same value however many make it, but a fold lands it once per unit.
            (UnitShare::Repeated, Write::Accumulate) => Drain::UnitZero,
            (UnitShare::Repeated, Write::Replace) | (UnitShare::Whole, _) => Drain::EachUnit,
        }
    }
}

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. It owns the [`UnitShare`], so cells the plane's
/// units hold partials of combine on commit and the contraction never asks.
///
/// It owns the [`Monoid`] and the [`CellRead`] for the same reason. Both are one fact about the
/// accumulation, not a fact about each cell, so they are settled where the view is built and read
/// from there by every seed and commit the leaf runs.
#[derive(CubeType)]
pub(crate) struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    units: UnitShare,
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
        values: MaskedMut<'a, Vector<E, V>, C>,
        #[comptime] units: UnitShare,
        #[comptime] split_share: SplitShare,
        #[comptime] write: Write,
        #[comptime] monoid: Monoid,
        #[comptime] init_from: InitFrom,
    ) -> Self {
        comptime!(write.admits(split_share, "AccumulateView"));
        AccumulateView::<'a, E, V, C> {
            values,
            units,
            monoid,
            cell_read: comptime!(CellRead::of(units, init_from, write)),
            drain: comptime!(Drain::of(units, write)),
        }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`].
    pub fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// How these cells are shared across the plane's units. A leaf that commits *conditionally*
    /// has to ask: past `Whole`, [`commit`](Self::commit) folds across the plane, and a plane op
    /// under divergent control flow is undefined.
    pub(crate) fn unit_share(&self) -> comptime_type!(UnitShare) {
        comptime!(self.units)
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

    /// Fold a finished block back. The fold reduces each `V`-wide cell element-wise and leaves
    /// every unit holding the total, so one writes: the plane's first unit where the whole plane
    /// shares one cell, each group's first unit where the plane carries a cell per group.
    pub fn commit(&mut self, pos: C, value: Vector<E, V>) {
        match comptime!(self.drain) {
            Drain::PlaneFold => {
                let combined = self.units.fold::<Vector<E, V>>(value, self.monoid);
                self.commit_shared(pos, combined, UNIT_POS_X == 0);
            }
            Drain::GroupFold { fold_mask } => {
                let combined = self.units.fold::<Vector<E, V>>(value, self.monoid);
                let unit_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                self.commit_shared(pos, combined, unit_in_group == 0);
            }
            // Nothing to combine, but a fold from units that repeat each other's work would land
            // once per unit, so one of them makes it.
            Drain::UnitZero => self.commit_shared(pos, value, UNIT_POS_X == 0),
            Drain::EachUnit => self.values.write(pos, value),
        }
    }

    /// Commit a cell the plane's units share, under the one unit elected to write it. Where this
    /// is the site that reads the cell, that unit folds it in, which no unit's seed could do.
    fn commit_shared(&mut self, pos: C, combined: Vector<E, V>, leader: bool) {
        if leader {
            match comptime!(self.cell_read) {
                CellRead::AtCommit => {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, self.monoid.combine::<Vector<E, V>>(old, combined));
                }
                CellRead::AtSeed | CellRead::Never => self.values.write(pos, combined),
            }
        }
    }
}
