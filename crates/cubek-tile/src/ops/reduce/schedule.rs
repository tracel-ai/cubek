//! The walks behind [`Tile::reduce_axis`](crate::Tile::reduce_axis): the plain recursion, and the
//! buffered one. The buffered schedule is [`pipelined_walk`]'s, shared with the matmul; this file
//! supplies only what a slot holds for a reduction and what consuming one computes.

use cubecl::prelude::*;

use super::kind::ReduceLeafKind;
use crate::*;

/// One level's reduction as a [`Pipelined`] operation: the accumulator it writes and the single
/// operand its slots stage.
#[derive(CubeType)]
pub(crate) struct ReduceWalk<Acc: Numeric, In: Numeric> {
    acc: Tile<Acc>,
    input: Tile<In>,
    #[cube(comptime)]
    inst: ReduceLeafKind,
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
        let has_plane_stage = ring.has_plane_stage();
        stage_walk_unrolled(
            &self.acc,
            comptime!(self.input.space.clone()),
            has_plane_stage,
        )
    }

    fn fill_fixed(&self, slot: &mut Staging<Tile<In>>, region: &Region) {
        slot.fill_fixed(&self.input, region);
    }

    fn fill_streamed(&self, slot: &mut Staging<Tile<In>>, region: &Region) {
        slot.fill_streamed(&self.input, region);
    }

    fn compute(
        &mut self,
        slot: &mut Staging<Tile<In>>,
        region: &Region,
        #[comptime] publish: bool,
    ) {
        let inst = comptime!(self.inst);
        if comptime!(publish) {
            slot.consume_final(|staged| {
                self.acc.at(region).reduce_axis(staged, inst);
            });
        } else {
            slot.consume(|staged| {
                self.acc.at(region).reduce_axis(staged, inst);
            });
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The input [`InPlace`](crate::Residence::InPlace): no slot, every read goes directly to where
    /// the operand already lives.
    pub(crate) fn reduce_direct<In: Numeric>(
        &mut self,
        input: &Tile<In>,
        #[comptime] inst: ReduceLeafKind,
        op_space: Space,
    ) {
        if self.tile_kind.static_level(comptime!(self.space.clone())) {
            let merged = comptime!({
                let merged = input.space.clone();
                assert!(
                    merged.is_static(),
                    "Tile::reduce_axis: a fragment output's walk unrolls over the operand merge, \
                     which must be static (a Dynamic extent cannot fold to the comptime \
                     coordinates fragment selection takes)"
                );
                merged
            });
            let walk =
                Walk::over_fastest(merged, comptime!(self.space.axis_at(self.space.rank() - 1)));
            for region in walk.unrolled() {
                self.at(&region).reduce_axis(&input.at(&region), inst);
            }
        } else {
            for region in Walk::over(op_space) {
                self.at(&region).reduce_axis(&input.at(&region), inst);
            }
        }
    }

    /// The level's regions through a ring of `depth` [`Staging`] slots: depth 1 fills a slot and
    /// consumes it per region, deeper rings overlap each region's fill with an earlier region's
    /// compute. [`pipelined_walk`] owns that schedule.
    pub(crate) fn reduce_buffered<In: Numeric>(
        &mut self,
        input: &Tile<In>,
        #[comptime] inst: ReduceLeafKind,
        op_space: Space,
        #[comptime] depth: usize,
    ) {
        let out = comptime!(self.space.clone());
        let mut walk = ReduceWalk::<Acc, In> {
            acc: self.clone(),
            input: input.clone(),
            inst,
        };
        pipelined_walk::<ReduceWalk<Acc, In>>(&mut walk, op_space, out, depth);
    }
}
