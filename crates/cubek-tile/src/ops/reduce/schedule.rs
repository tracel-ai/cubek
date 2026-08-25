//! The walk behind [`Tile::reduce_axis`](crate::Tile::reduce_axis). The schedule is
//! [`pipelined_walk`]'s, shared with the matmul; this file supplies only what a slot holds for a
//! reduction and what consuming one computes.

use cubecl::prelude::*;

use crate::*;

/// One level's reduction as a [`Pipelined`] operation: the accumulator it writes and the single
/// operand its slots stage.
#[derive(CubeType)]
pub(crate) struct ReduceWalk<Acc: Numeric, In: Numeric> {
    acc: Tile<Acc>,
    input: Tile<In>,
    #[cube(comptime)]
    monoid: Monoid,
}

#[cube]
impl<Acc: Numeric, In: Numeric> Pipelined for ReduceWalk<Acc, In> {
    type Slot = Tile<In>;

    fn ring(
        &self,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<Tile<In>> {
        Ring::unary(&self.input, op_space, out, depth)
    }

    /// [`stage_walk_unrolled`] over the sole operand's space, which is the whole merge here.
    fn unrolled(&self, ring: &Ring<Tile<In>>) -> comptime_type!(bool) {
        let has_fragment_read = ring.has_fragment_read();
        stage_walk_unrolled(
            &self.acc,
            comptime!(self.input.space.clone()),
            has_fragment_read,
        )
    }

    fn fill_fixed(&self, slot: &mut Staging<Tile<In>>, region: &Region) {
        slot.fill_fixed(&self.input, region);
    }

    fn fill_streamed(&self, slot: &mut Staging<Tile<In>>, region: &Region) {
        slot.fill_streamed(&self.input, region);
    }

    /// Fold `region` out of the slot, [`read_operand`] bringing the payload to it.
    fn compute(
        &mut self,
        slot: &mut Staging<Tile<In>>,
        region: &Region,
        #[comptime] publish: bool,
    ) {
        let monoid = comptime!(self.monoid);
        let plan = slot.plan(LHS);
        let payload = comptime!(plan.payload);
        if comptime!(publish) {
            slot.publish();
        }
        slot.consume(|staged| {
            let input = read_operand(staged, region, payload);
            self.acc.at(region).reduce_axis(&input, monoid);
        });
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The level's regions through a ring of `depth` [`Staging`] slots: depth 1 fills a slot and
    /// consumes it per region, deeper rings overlap each region's fill with an earlier region's
    /// compute. [`pipelined_walk`] owns that schedule.
    pub(crate) fn reduce_buffered<In: Numeric>(
        &mut self,
        input: &Tile<In>,
        #[comptime] monoid: Monoid,
        op_space: Space,
        #[comptime] depth: usize,
    ) {
        let out = comptime!(self.space.clone());
        let mut walk = ReduceWalk::<Acc, In> {
            acc: self.clone(),
            input: input.clone(),
            monoid,
        };
        pipelined_walk::<ReduceWalk<Acc, In>>(&mut walk, op_space, out, depth);
    }
}
