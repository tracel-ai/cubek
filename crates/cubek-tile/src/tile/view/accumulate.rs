use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::{instruction::plane, *};

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. The write-side mirror of a
/// quantized view dequantizing on read: it owns the [`LaneShare`], so cells the plane's lanes
/// hold partials of combine on commit and the contraction never asks.
///
/// It owns the [`Monoid`] for the same reason. The algebra is one fact about the accumulation,
/// not a fact about each cell, so it is stated where the view is built and read from there by
/// every seed and commit the leaf runs.
#[derive(CubeType)]
pub struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedViewMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    lane_share: LaneShare,
    #[cube(comptime)]
    monoid: Monoid,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedViewMut<'a, Vector<E, V>, C>,
        #[comptime] lane_share: LaneShare,
        #[comptime] monoid: Monoid,
    ) -> Self {
        AccumulateView::<'a, E, V, C> {
            values,
            lane_share,
            monoid,
        }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`].
    pub fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// Whether a non-empty output block is wholly valid for unchecked seed/commit accesses.
    pub fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        self.values.block_in_bounds(pos, extent)
    }

    /// A block's starting value. A partial starts at the monoid's identity: the shared cell is
    /// folded in once, by the lane that commits, so seeding from it would count it once per lane.
    /// Only `LaneShare::Whole`, which holds the cell outright, seeds from it.
    pub fn seed(&self, pos: C) -> Vector<E, V> {
        match comptime!(self.lane_share) {
            LaneShare::Plane | LaneShare::Group { .. } => {
                Vector::<E, V>::cast_from(Monoid::identity::<E>(self.monoid))
            }
            LaneShare::Whole => self.values.read(pos),
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
                if UNIT_POS_X == 0 {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, self.monoid.fold::<Vector<E, V>>(old, combined));
                }
            }
            LaneShare::Group { fold_mask } => {
                let combined = plane::group(value, fold_mask, self.monoid);
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                if lane_in_group == 0 {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, self.monoid.fold::<Vector<E, V>>(old, combined));
                }
            }
            LaneShare::Whole => self.values.write(pos, value),
        }
    }
}
