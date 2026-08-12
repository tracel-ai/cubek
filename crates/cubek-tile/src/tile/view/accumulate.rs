use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::{
    instruction::{max as inst_max, min as inst_min, sum as inst_sum},
    ops::ReduceLeafKind,
    *,
};

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. The write-side mirror of a
/// quantized view dequantizing on read: it owns the [`LaneShare`], so cells the plane's lanes
/// hold partials of combine on commit and the contraction never asks.
#[derive(CubeType)]
pub struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedViewMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    lane_share: LaneShare,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedViewMut<'a, Vector<E, V>, C>,
        #[comptime] lane_share: LaneShare,
    ) -> Self {
        AccumulateView::<'a, E, V, C> { values, lane_share }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`].
    pub fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// A block's starting value. A partial starts at zero: the shared cell is folded in once, by
    /// the lane that commits, so seeding from it would count it once per lane.
    pub fn seed(&self, pos: C) -> Vector<E, V> {
        self.seed_reduce(pos, comptime!(ReduceLeafKind::Sum))
    }

    /// Seed the accumulator with the appropriate identity element for `inst` when folding across
    /// lanes (`LaneShare::Plane` or `LaneShare::Group`), or with the existing accumulator value
    /// under `LaneShare::Whole`.
    pub fn seed_reduce(&self, pos: C, #[comptime] inst: ReduceLeafKind) -> Vector<E, V> {
        match comptime!(self.lane_share) {
            LaneShare::Plane | LaneShare::Group { .. } => match comptime!(inst) {
                ReduceLeafKind::Sum => Vector::<E, V>::cast_from(E::from_int(0)),
                ReduceLeafKind::Max => Vector::<E, V>::cast_from(E::min_value()),
                ReduceLeafKind::Min => Vector::<E, V>::cast_from(E::max_value()),
            },
            LaneShare::Whole => self.values.read(pos),
        }
    }

    /// Fold a finished block back. The fold reduces each `V`-wide cell element-wise and leaves
    /// every lane holding a partial of it with the total, so one of them writes and its siblings
    /// don't all hit the address: the plane's first lane where the whole plane shares one cell,
    /// each group's first lane where the plane carries a cell per group.
    pub fn commit(&mut self, pos: C, value: Vector<E, V>) {
        self.commit_reduce(pos, value, comptime!(ReduceLeafKind::Sum));
    }

    /// Fold a finished block back according to `inst` (`Sum`, `Max`, or `Min`).
    pub fn commit_reduce(&mut self, pos: C, value: Vector<E, V>, #[comptime] inst: ReduceLeafKind) {
        match comptime!(self.lane_share) {
            LaneShare::Plane => {
                let combined = match comptime!(inst) {
                    ReduceLeafKind::Sum => plane_sum(value),
                    ReduceLeafKind::Max => plane_max(value),
                    ReduceLeafKind::Min => plane_min(value),
                };
                if UNIT_POS_X == 0 {
                    let old = self.values.read(pos.clone());
                    let result = match comptime!(inst) {
                        ReduceLeafKind::Sum => old + combined,
                        ReduceLeafKind::Max => max(old, combined),
                        ReduceLeafKind::Min => min(old, combined),
                    };
                    self.values.write(pos, result);
                }
            }
            LaneShare::Group { fold_mask } => {
                let combined = match comptime!(inst) {
                    ReduceLeafKind::Sum => inst_sum::group::<E, V>(value, fold_mask),
                    ReduceLeafKind::Max => inst_max::group::<E, V>(value, fold_mask),
                    ReduceLeafKind::Min => inst_min::group::<E, V>(value, fold_mask),
                };
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                if lane_in_group == 0 {
                    let old = self.values.read(pos.clone());
                    let result = match comptime!(inst) {
                        ReduceLeafKind::Sum => old + combined,
                        ReduceLeafKind::Max => max(old, combined),
                        ReduceLeafKind::Min => min(old, combined),
                    };
                    self.values.write(pos, result);
                }
            }
            LaneShare::Whole => self.values.write(pos, value),
        }
    }
}
