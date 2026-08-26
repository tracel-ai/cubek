use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::{instruction::plane, *};

/// What an accumulation starts from: the statement a verb makes about the cells it is about to
/// write ([`Tile::mm`] and [`Tile::reduce_axis`] say [`Identity`](InitFrom::Identity), the
/// accumulating verbs say [`Cell`](InitFrom::Cell)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InitFrom {
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
    const fn of(lane_share: LaneShare, init_from: InitFrom) -> Self {
        match init_from {
            InitFrom::Identity => CellRead::Never,
            InitFrom::Cell => match lane_share {
                LaneShare::Whole => CellRead::AtSeed,
                LaneShare::Plane | LaneShare::Group { .. } => CellRead::AtCommit,
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
pub struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedViewMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    lane_share: LaneShare,
    #[cube(comptime)]
    monoid: Monoid,
    #[cube(comptime)]
    cell_read: CellRead,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedViewMut<'a, Vector<E, V>, C>,
        #[comptime] lane_share: LaneShare,
        #[comptime] monoid: Monoid,
        #[comptime] init_from: InitFrom,
    ) -> Self {
        AccumulateView::<'a, E, V, C> {
            values,
            lane_share,
            monoid,
            cell_read: comptime!(CellRead::of(lane_share, init_from)),
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
    pub fn lane_share(&self) -> comptime_type!(LaneShare) {
        comptime!(self.lane_share)
    }

    /// The monoid these cells fold under, stated where the view was built. A register block asks
    /// so its own seed and commit start from and collapse under the same fold this does.
    pub fn monoid(&self) -> comptime_type!(Monoid) {
        comptime!(self.monoid)
    }

    /// Whether a non-empty output block is wholly valid for unchecked seed/commit accesses.
    pub fn block_in_bounds(&self, pos: C, extent: C) -> bool {
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
        match comptime!(self.lane_share) {
            LaneShare::Plane => {
                let combined = plane::broadcast::<Vector<E, V>>(value, self.monoid);
                self.commit_shared(pos, combined, UNIT_POS_X == 0);
            }
            LaneShare::Group { fold_mask } => {
                let combined = plane::group(value, fold_mask, self.monoid);
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                self.commit_shared(pos, combined, lane_in_group == 0);
            }
            LaneShare::Whole => self.values.write(pos, value),
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
