//! The walk behind [`Tile::mma`](crate::Tile::mma). A level's structure is [`pipelined_walk`]'s
//! whatever its [`Buffering`](crate::Buffering) depth and wherever its operands live; this file
//! supplies only what a slot holds for a matmul and what consuming one computes.

use cubecl::prelude::*;

use crate::*;

/// One level's matmul as a [`Pipelined`] operation: the accumulator it writes and the two operands
/// its slots stage. The tiles are handles, so the walk addressing them addresses the same storage
/// the caller passed in.
#[derive(CubeType)]
pub(crate) struct MmaWalk<Acc: Numeric, Lhs: Numeric, Rhs: Numeric> {
    acc: Tile<Acc>,
    lhs: Tile<Lhs>,
    rhs: Tile<Rhs>,
}

#[cube]
impl<Acc: Numeric, Lhs: Numeric, Rhs: Numeric> Pipelined for MmaWalk<Acc, Lhs, Rhs> {
    type Slot = (Tile<Lhs>, Tile<Rhs>);

    fn ring(
        &self,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<(Tile<Lhs>, Tile<Rhs>)> {
        Ring::binary(&self.lhs, &self.rhs, op_space, out, depth)
    }

    /// [`stage_walk_unrolled`] over the two operands' merge, which is the space a fragment read
    /// has to be static-walkable in.
    fn unrolled(&self, ring: &Ring<(Tile<Lhs>, Tile<Rhs>)>) -> comptime_type!(bool) {
        let has_fragment_read = ring.has_fragment_read();
        stage_walk_unrolled(
            &self.acc,
            comptime!(Space::merge(&[&self.lhs.space, &self.rhs.space])),
            has_fragment_read,
        )
    }

    fn fill_fixed(&self, slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>, region: &Region) {
        slot.fill_fixed(&self.lhs, &self.rhs, region);
    }

    fn fill_streamed(&self, slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>, region: &Region) {
        slot.fill_streamed(&self.lhs, &self.rhs, region);
    }

    /// Contract `region` out of the slot, [`read_operand`] bringing each payload to it.
    fn compute(
        &mut self,
        slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>,
        region: &Region,
        #[comptime] publish: bool,
    ) {
        let lhs_plan = slot.plan(LHS);
        let rhs_plan = slot.plan(RHS);
        let lhs_payload = comptime!(lhs_plan.payload);
        let rhs_payload = comptime!(rhs_plan.payload);
        if comptime!(publish) {
            slot.publish();
        }
        slot.consume(|staged_lhs, staged_rhs| {
            let lhs = read_operand(staged_lhs, region, lhs_payload);
            let rhs = read_operand(staged_rhs, region, rhs_payload);
            self.acc.at(region).mma(&lhs, &rhs)
        });
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The level's regions through a ring of `depth` [`Staging`] slots: depth 1 fills a slot and
    /// consumes it per region, deeper rings overlap each region's fill with an earlier region's
    /// compute. [`pipelined_walk`] owns that schedule.
    pub(crate) fn mma_buffered<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        op_space: Space,
        #[comptime] depth: usize,
    ) {
        let out = comptime!(self.space.clone());
        let mut walk = MmaWalk::<Acc, Lhs, Rhs> {
            acc: self.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        };
        pipelined_walk::<MmaWalk<Acc, Lhs, Rhs>>(&mut walk, op_space, out, depth);
    }
}
