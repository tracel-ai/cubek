use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, Coords2d},
};

use crate::{instruction::plane, *};

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back. The write-side mirror of a
/// quantized view dequantizing on read: it owns the [`LaneShare`], so cells the plane's lanes
/// hold partials of combine on commit and the contraction never asks.
#[derive(CubeType)]
pub struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedViewMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    lane_share: LaneShare,
    #[cube(comptime)]
    overwrites: bool,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedViewMut<'a, Vector<E, V>, C>,
        #[comptime] lane_share: LaneShare,
        #[comptime] overwrites: bool,
    ) -> Self {
        AccumulateView::<'a, E, V, C> {
            values,
            lane_share,
            overwrites,
        }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain view.
    pub fn check(&self) -> comptime_type!(bool) {
        self.values.check
    }

    /// Whether a whole block starting at `pos` is in bounds.
    pub fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        self.values.block_in_bounds(pos, extent)
    }

    /// A block's starting value for a `fold` fold. Two reasons to start from `fold`'s identity
    /// rather than the cell: a partial holds no whole cell to carry forward (the shared cell is
    /// folded in once, by the lane that commits, so starting from it would count it once per
    /// lane), and an overwriting accumulation is not folding onto the cell at all.
    pub fn seed(&self, pos: C, #[comptime] fold: LeafOp) -> Vector<E, V> {
        let reads_back = self.reads_back();
        if comptime!(reads_back) {
            self.values.read(pos)
        } else {
            Vector::<E, V>::cast_from(LeafOp::identity::<E>(fold))
        }
    }

    /// Fold `value` into the cell at `pos`. A lane-shared cell is combined across the lanes that
    /// hold it first, then written by one; a whole cell is the lane's own.
    pub fn commit(&mut self, pos: C, value: Vector<E, V>, #[comptime] fold: LeafOp) {
        match comptime!(self.lane_share) {
            LaneShare::Plane => {
                let combined = plane::broadcast::<Vector<E, V>>(value, fold);
                self.commit_shared(pos, combined, UNIT_POS_X == 0, fold);
            }
            LaneShare::Group { fold_mask } => {
                let combined = plane::group(value, fold_mask, fold);
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                self.commit_shared(pos, combined, lane_in_group == 0, fold);
            }
            LaneShare::Whole => self.values.write(pos, value),
        }
    }

    /// Commit a cell combined across the lanes that share it, under the elected lane. It folds
    /// the cell's own value in exactly once, unless the accumulation overwrites it, in which case
    /// there is nothing to fold: the block never started from it, and it holds whatever it held
    /// before the operation.
    fn commit_shared(
        &mut self,
        pos: C,
        combined: Vector<E, V>,
        leader: bool,
        #[comptime] fold: LeafOp,
    ) {
        let reads_back = self.reads_back();
        if leader {
            if comptime!(reads_back) {
                let old = self.values.read(pos.clone());
                self.values
                    .write(pos, LeafOp::combine::<Vector<E, V>>(old, combined, fold));
            } else {
                self.values.write(pos, combined);
            }
        }
    }

    /// Whether the cell's own value is read back into the result: it is a whole cell this lane
    /// owns, and the accumulation folds onto it rather than overwriting it.
    fn reads_back(&self) -> comptime_type!(bool) {
        comptime!(matches!(self.lane_share, LaneShare::Whole) && !self.overwrites)
    }
}
