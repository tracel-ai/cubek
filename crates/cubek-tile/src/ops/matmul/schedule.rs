//! The walk behind [`Tile::mma`](crate::Tile::mma). A level's structure is [`pipelined_walk`]'s
//! whatever its [`Buffering`](crate::Buffering) depth and wherever its operands live; this file
//! supplies only what a slot holds for a matmul and what consuming one computes.

use cubecl::prelude::*;

use crate::*;

/// One level's matmul as a [`Pipelined`] operation: the accumulator it writes, the two operands
/// its slots stage, and the algebra they contract under. The tiles are handles, so the walk
/// addressing them addresses the same storage the caller passed in.
#[derive(CubeType)]
pub(crate) struct MmaWalk<Acc: Numeric, Lhs: Numeric, Rhs: Numeric> {
    acc: Tile<Acc>,
    lhs: Tile<Lhs>,
    rhs: Tile<Rhs>,
    #[cube(comptime)]
    semiring: Semiring,
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
            self.acc
                .at(region)
                .mma(&lhs, &rhs, comptime!(self.semiring))
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
        #[comptime] semiring: Semiring,
    ) {
        let out = comptime!(self.space.clone());
        let mut walk = MmaWalk::<Acc, Lhs, Rhs> {
            acc: self.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            semiring,
        };
        pipelined_walk::<MmaWalk<Acc, Lhs, Rhs>>(&mut walk, op_space, out, depth);
    }
}

/// [`MmaWalk`] with scales riding beside the ring.
///
/// Beside it, not in it. The ring stages what the walk reuses, and a scale is one value over a whole
/// block: cache-served wherever it lies, and staging one materializes the expansion that reading it
/// coarsely exists to avoid. There is also nothing to single out. The scales are a list, and a ring
/// that staged one member of it would be choosing arbitrarily.
///
/// A level that states a residence the ring cannot give it is refused at the bind rather than filled
/// and ignored ([`StridedTileSource::residence`](crate::StridedTileSource::residence)).
#[derive(CubeType)]
pub(crate) struct MmaScaledWalk<Acc: Numeric, Lhs: Numeric, Rhs: Numeric, S: Numeric> {
    acc: Tile<Acc>,
    lhs: Tile<Lhs>,
    rhs: Tile<Rhs>,
    scales: Sequence<Tile<S>>,
    #[cube(comptime)]
    semiring: Semiring,
}

#[cube]
impl<Acc: Numeric, Lhs: Numeric, Rhs: Numeric, S: Numeric> Pipelined
    for MmaScaledWalk<Acc, Lhs, Rhs, S>
{
    type Slot = (Tile<Lhs>, Tile<Rhs>);

    fn ring(
        &self,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<(Tile<Lhs>, Tile<Rhs>)> {
        Ring::binary(&self.lhs, &self.rhs, op_space, out, depth)
    }

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
        let all = self.scales.clone();
        if comptime!(publish) {
            slot.publish();
        }
        slot.consume(|staged_lhs, staged_rhs| {
            let lhs = read_operand(staged_lhs, region, lhs_payload);
            let rhs = read_operand(staged_rhs, region, rhs_payload);
            let mut scales = Sequence::new();
            #[unroll]
            for k in 0..all.len() {
                scales.push(all.index(k).at(region));
            }
            self.acc
                .at(region)
                .mma_scaled(&lhs, &rhs, &scales, comptime!(self.semiring))
        });
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// [`mma_buffered`](Tile::mma_buffered) with the scales carried alongside.
    pub(crate) fn mma_scaled_buffered<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        op_space: Space,
        #[comptime] depth: usize,
        #[comptime] semiring: Semiring,
    ) {
        let out = comptime!(self.space.clone());
        // A scale is one value over a whole block, so the ring has nothing to amortize by staging
        // one and does not carry them. Saying otherwise is refused rather than ignored, which is
        // the difference between a plan that is wrong and one that is quietly not followed.
        #[unroll]
        for k in 0..scales.len() {
            let residence = scales.index(k).residence(comptime!(&out));
            comptime!(assert!(
                residence == Residence::InPlace,
                "mma_scaled: a scale level states {residence:?}, and scales are read where they \
                 lie. One value covers a whole block, so a stage would hold a copy nothing reads \
                 more than once"
            ));
        }
        let mut walk = MmaScaledWalk::<Acc, Lhs, Rhs, S> {
            acc: self.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            scales: scales.clone(),
            semiring,
        };
        pipelined_walk::<MmaScaledWalk<Acc, Lhs, Rhs, S>>(&mut walk, op_space, out, depth);
    }
}
