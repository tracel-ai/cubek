//! The walks behind [`Tile::mma`](crate::Tile::mma): the plain recursion, and the buffered one.
//! A buffered walk's structure is [`pipelined_walk`]'s whatever its
//! [`Buffering`](crate::Buffering) depth; this file supplies only what a slot holds for a matmul
//! and what consuming one computes.

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

    /// [`stage_walk_unrolled`] over the two operands' merge, which is the space a plane stage has
    /// to be static-walkable in.
    fn unrolled(&self, ring: &Ring<(Tile<Lhs>, Tile<Rhs>)>) -> comptime_type!(bool) {
        let has_plane_stage = ring.has_plane_stage();
        stage_walk_unrolled(
            &self.acc,
            comptime!(Space::merge(&[&self.lhs.space, &self.rhs.space])),
            has_plane_stage,
        )
    }

    fn fill_fixed(&self, slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>, region: &Region) {
        slot.fill_fixed(&self.lhs, &self.rhs, region);
    }

    fn fill_streamed(&self, slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>, region: &Region) {
        slot.fill_streamed(&self.lhs, &self.rhs, region);
    }

    fn compute(
        &mut self,
        slot: &mut Staging<(Tile<Lhs>, Tile<Rhs>)>,
        region: &Region,
        #[comptime] publish: bool,
    ) {
        if comptime!(publish) {
            slot.consume_final(|staged_lhs, staged_rhs| {
                self.acc.at(region).mma(staged_lhs, staged_rhs)
            });
        } else {
            slot.consume(|staged_lhs, staged_rhs| self.acc.at(region).mma(staged_lhs, staged_rhs));
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Every operand [`InPlace`](crate::Residence::InPlace): no slot, every read goes to where the
    /// operand already lives. A fragment output demands the unrolled walk (its coordinates fold to
    /// constants, which select fragments); a memory output keeps the compact runtime loop.
    pub(crate) fn mma_direct<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        op_space: Space,
    ) {
        if self.tile_kind.static_level(comptime!(self.space.clone())) {
            let merged = comptime!({
                let merged = Space::merge(&[&lhs.space, &rhs.space]);
                assert!(
                    merged.is_static(),
                    "Tile::mma: a fragment output's walk unrolls over the operand merge, \
                     which must be static (a Dynamic extent cannot fold to the comptime \
                     coordinates fragment selection takes)"
                );
                merged
            });
            let walk =
                Walk::over_fastest(merged, comptime!(self.space.axis_at(self.space.rank() - 2)));
            for region in walk.unrolled() {
                self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
            }
        } else {
            for region in Walk::over(op_space) {
                self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
            }
        }
    }

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
