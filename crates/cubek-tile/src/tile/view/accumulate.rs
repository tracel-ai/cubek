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

    /// Whether a non-empty output block is wholly valid for unchecked seed/commit accesses.
    pub fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        self.values.block_in_bounds(pos, extent)
    }

    /// A block's starting value for an `inst` fold. A partial starts at `inst`'s identity: the
    /// shared cell is folded in once, by the lane that commits, so seeding from it would count it
    /// once per lane. Only `LaneShare::Whole`, which holds the cell outright, seeds from it.
    pub fn seed(
        &self,
        pos: C,
        #[comptime] inst: LeafOp,
        #[comptime] replace: bool,
    ) -> Vector<E, V> {
        // A folded share holds no whole cell to carry forward, and a replacing contraction is
        // discarding whatever is there anyway: both start from the identity and skip the read.
        let carries = comptime!(matches!(self.lane_share, LaneShare::Whole) && !replace);
        if comptime!(carries) {
            self.values.read(pos)
        } else {
            Vector::<E, V>::cast_from(LeafOp::identity::<E>(inst))
        }
    }

    /// Fold a finished block back under `inst`. The fold reduces each `V`-wide cell element-wise
    /// and leaves every lane holding a partial of it with the total, so one of them writes and its
    /// siblings don't all hit the address: the plane's first lane where the whole plane shares one
    /// cell, each group's first lane where the plane carries a cell per group.
    pub fn commit(&mut self, pos: C, value: Vector<E, V>, #[comptime] inst: LeafOp) {
        match comptime!(self.lane_share) {
            LaneShare::Plane => {
                let combined = plane::broadcast::<Vector<E, V>>(value, inst);
                if UNIT_POS_X == 0 {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, LeafOp::combine::<Vector<E, V>>(old, combined, inst));
                }
            }
            LaneShare::Group { fold_mask } => {
                let combined = plane::group(value, fold_mask, inst);
                let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                if lane_in_group == 0 {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, LeafOp::combine::<Vector<E, V>>(old, combined, inst));
                }
            }
            LaneShare::Whole => self.values.write(pos, value),
        }
    }
}

/// How a contraction's `mr × nr` block reaches its accumulator's memory.
///
/// The block is `V` wide because the rhs is ([`contract::memory`](crate::microkernel)), and the
/// sink usually holds lines of exactly that width, so a cell is one access. It does not have to.
/// An axis global memory cannot line up has no line view at any index (an `NHWC` tensor at
/// `C = 3` starts every row at element `3r`, so no 4-wide line ever lands on a row boundary), yet
/// the operand feeding it can still be *padded* into lines once it reaches shared memory
/// ([`StagePlan::width`](crate::StagePlan)). The contraction then runs `V` wide against a sink
/// addressed one scalar at a time, and this is the join between them: the block's column `n` is
/// the sink's columns `n·V … n·V + V - 1`.
///
/// The padding lanes need no handling here. The sink's innermost edge is the padded one while its
/// buffer is not, so the columns past the real extent are the window's ordinary overhang and its
/// mask drops them, exactly as it drops a partial tile on any other axis.
#[derive(CubeType)]
pub struct BlockAccumulate<'a, E: Numeric, V: Size> {
    cells: BlockCells<'a, E, V>,
    /// The block's line width, `V` as a number. Only [`Lanes`](BlockCells::Lanes) reads it.
    #[cube(comptime)]
    width: usize,
}

/// Which of the two addressings a [`BlockAccumulate`] holds.
#[derive(CubeType)]
pub enum BlockCells<'a, E: Numeric, V: Size> {
    /// The sink's line is the block's: one access per cell, and the lane arithmetic is dead.
    Lines(AccumulateView<'a, E, V, Coords2d>),
    /// The sink is scalar and the block is `V` wide: `V` accesses per cell.
    Lanes(AccumulateView<'a, E, Const<1>, Coords2d>),
}

#[cube]
impl<'a, E: Numeric, V: Size> BlockAccumulate<'a, E, V> {
    pub fn new(cells: BlockCells<'a, E, V>, #[comptime] width: usize) -> Self {
        BlockAccumulate::<'a, E, V> { cells, width }
    }

    /// The underlying overhang-mask flag, so a leaf makes the same unroll decision it makes on a
    /// plain [`MatrixView`](crate::MatrixView).
    pub fn check(&self) -> comptime_type!(bool) {
        match &self.cells {
            BlockCells::Lines(v) => v.check(),
            BlockCells::Lanes(v) => v.check(),
        }
    }

    /// Whether the `rows × cols` block at `(row, col)` is wholly valid, so seed and commit can
    /// skip their masks. Stated in block cells; a scalar sink spans `width` of its own columns per
    /// cell, so its question is asked over the widened span.
    pub fn block_in_bounds(&self, row: u32, col: u32, rows: u32, cols: u32) -> bool {
        match &self.cells {
            BlockCells::Lines(v) => v.block_in_bounds((row, col), (rows, cols)),
            BlockCells::Lanes(v) => {
                let w = comptime!(self.width as u32);
                v.block_in_bounds((row, col.fmul(w)), (rows, cols.fmul(w)))
            }
        }
    }

    /// The block cell at `(row, col)`'s starting value for an `inst` fold. See
    /// [`AccumulateView::seed`].
    pub fn seed(
        &self,
        row: u32,
        col: u32,
        #[comptime] inst: LeafOp,
        #[comptime] replace: bool,
    ) -> Vector<E, V> {
        match &self.cells {
            BlockCells::Lines(v) => v.seed((row, col), inst, replace),
            BlockCells::Lanes(v) => {
                let base = col.fmul(comptime!(self.width as u32));
                let mut out = Vector::<E, V>::cast_from(E::from_int(0));
                #[unroll]
                for l in 0..comptime!(self.width) {
                    out.insert(
                        l,
                        v.seed((row, base.fadd(comptime!(l as u32))), inst, replace)
                            .extract(0usize),
                    );
                }
                out
            }
        }
    }

    /// Fold the finished block cell at `(row, col)` back under `inst`. See
    /// [`AccumulateView::commit`].
    pub fn commit(&mut self, row: u32, col: u32, value: Vector<E, V>, #[comptime] inst: LeafOp) {
        let width = comptime!(self.width);
        match &mut self.cells {
            BlockCells::Lines(v) => v.commit((row, col), value, inst),
            BlockCells::Lanes(v) => {
                let base = col.fmul(comptime!(width as u32));
                #[unroll]
                for l in 0..width {
                    v.commit(
                        (row, base.fadd(comptime!(l as u32))),
                        Vector::<E, Const<1>>::cast_from(value.extract(l)),
                        inst,
                    );
                }
            }
        }
    }
}
