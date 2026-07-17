use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. The write-side mirror of a
/// [`QuantizedView`] dequantizing on read: it owns the [`LaneShare`], so cells the plane's lanes
/// hold partials of combine on commit and the contraction never asks.
#[derive(CubeType)]
pub struct AccumulateView<'a, E: Numeric, V: Size> {
    values: MatrixViewMut<'a, Vector<E, V>>,
    #[cube(comptime)]
    lane_share: LaneShare,
}

#[cube]
impl<'a, E: Numeric, V: Size> AccumulateView<'a, E, V> {
    pub(crate) fn new(
        values: MatrixViewMut<'a, Vector<E, V>>,
        #[comptime] lane_share: LaneShare,
    ) -> Self {
        AccumulateView::<'a, E, V> { values, lane_share }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`].
    pub fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// A block's starting value. A partial starts at zero: the shared cell is folded in once, by
    /// the lane that commits, so seeding from it would count it once per lane.
    pub fn seed(&self, pos: Coords2d) -> Vector<E, V> {
        match comptime!(self.lane_share) {
            LaneShare::Partial => Vector::<E, V>::cast_from(E::from_int(0)),
            LaneShare::Whole => self.values.read(pos),
        }
    }

    /// Fold a finished block back. `plane_sum` reduces each `V`-wide cell element-wise, leaving
    /// every lane holding the total, so one lane writes and siblings don't all hit the address.
    pub fn commit(&mut self, pos: Coords2d, value: Vector<E, V>) {
        match comptime!(self.lane_share) {
            LaneShare::Partial => {
                let combined = plane_sum(value);
                if UNIT_POS_X == 0 {
                    self.values.write(pos, self.values.read(pos) + combined);
                }
            }
            LaneShare::Whole => self.values.write(pos, value),
        }
    }
}
