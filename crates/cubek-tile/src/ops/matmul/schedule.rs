//! The walks behind [`Tile::mma`](crate::Tile::mma), one per [`Schedule`]. A schedule's body
//! is pure structure; kind decisions (slot store, rendezvous, fill dispatch) are
//! delegated, chiefly to [`Staging::new`].

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `Direct`: no staging, every read goes to where the operand lives. A fragment
    /// output demands the unrolled walk (its coordinates fold to constants, which
    /// select fragments); a memory output keeps the compact runtime loop.
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

    /// `Staged`: per region, fill a [`Staging`] slot with the operands and consume it into
    /// the recursion. An operand the walk leaves unchanged (its space lacks every walked axis,
    /// the same structural fact as broadcast omission) fills once, above the loop; re-filling
    /// per region would just move the same window again. `consume_final` every region, since no
    /// later fill publishes within an iteration.
    ///
    /// The walk unrolls when the level *cuts* a fragment-partition output (each region selects
    /// its block by comptime coordinate) or on a static register-staged level (comptime regions
    /// land window offsets as immediates). An smem-staged level stays rolled: unrolling would
    /// re-stage its shared memory per copy.
    pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        op_space: Space,
    ) {
        // `Staging` decides which operand (if any) is pinned: walk-invariant, so its window
        // never moves and it fills once, above the loop. The rest stream, refilled per region.
        let mut staging = Staging::new(
            lhs,
            rhs,
            comptime!(op_space.clone()),
            comptime!(self.space.clone()),
        );
        let cuts = self.tile_kind.cuts_partition(comptime!(self.space.clone()));
        // A plane stage selects its tiles by comptime coordinate, so it stands up only under an
        // unrolled walk, and only when the operand merge is itself static-walkable.
        let stage = staging.stage();
        let plane_stage = comptime!(
            stage == OperandStage::Plane
                && Space::merge(&[&lhs.space, &rhs.space]).static_walkable()
        );
        let unroll = comptime!(cuts || plane_stage);

        let walk = Walk::over(op_space);
        staging.fill_pinned(lhs, rhs, &walk.region(0));
        let walk = if comptime!(unroll) {
            walk.unrolled()
        } else {
            walk
        };
        for region in walk {
            staging.fill_streamed(lhs, rhs, &region);
            staging.consume_final(|staged_lhs, staged_rhs| {
                self.at(&region).mma(staged_lhs, staged_rhs)
            });
        }
    }

    /// `DoubleBuffered`: two [`Staging`] slots driven `fill`/`consume` on alternating
    /// regions so one slot's fill overlaps the other's compute.
    pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        op_space: Space,
    ) {
        // Keep this region-index protocol in lockstep with `Tile::reduce_double` in
        // `ops/reduce/schedule.rs`: only the fill/consume bodies differ by operand arity.
        // Changes to prologue, alternating prefetch, or epilogue handling must be made in both.
        // Double-buffering fills both operands every region (see the raw `fill`s below), so the
        // pin flags go unread; pass the operation space only to satisfy `new`.
        let mut even_slot = Staging::new(
            lhs,
            rhs,
            comptime!(op_space.clone()),
            comptime!(self.space.clone()),
        );
        let mut odd_slot = Staging::new(
            lhs,
            rhs,
            comptime!(op_space.clone()),
            comptime!(self.space.clone()),
        );

        // Double-buffering needs random access (prefetch the next region), so it indexes the
        // `walk` by hand rather than iterating.
        let walk = Walk::over(op_space);
        let n = walk.total();

        // Prologue: prime the even slot with region 0.
        let first = walk.region(0);
        even_slot.fill(|staged_operands, pipe| {
            pipe.fill(&mut staged_operands.0, &lhs.at(&first));
            pipe.fill(&mut staged_operands.1, &rhs.at(&first));
        });

        for p in 0..n / 2 {
            let even = p * 2;
            let odd = even + 1;

            // Prefetch the odd region into its slot (its fill overlaps the compute below), then
            // compute the even region from the even slot.
            let odd_region = walk.region(odd);
            odd_slot.fill(|staged_operands, pipe| {
                pipe.fill(&mut staged_operands.0, &lhs.at(&odd_region));
                pipe.fill(&mut staged_operands.1, &rhs.at(&odd_region));
            });
            let even_region = walk.region(even);
            even_slot.consume(|staged_lhs, staged_rhs| {
                self.at(&even_region).mma(staged_lhs, staged_rhs)
            });

            // Prefetch the next even region back into the even slot (if it exists), then compute
            // the odd region from the odd slot; on the walk's final region no fill follows, so
            // `consume_final` publishes the odd slot itself.
            if odd + 1 < n {
                let next_even = walk.region(odd + 1);
                even_slot.fill(|staged_operands, pipe| {
                    pipe.fill(&mut staged_operands.0, &lhs.at(&next_even));
                    pipe.fill(&mut staged_operands.1, &rhs.at(&next_even));
                });
                odd_slot.consume(|staged_lhs, staged_rhs| {
                    self.at(&odd_region).mma(staged_lhs, staged_rhs)
                });
            } else {
                odd_slot.consume_final(|staged_lhs, staged_rhs| {
                    self.at(&odd_region).mma(staged_lhs, staged_rhs)
                });
            }
        }

        // An odd total leaves the last region primed in the even slot with no consumer in the
        // loop; no fill follows, so `consume_final` publishes it.
        if n % 2 == 1 {
            let last = walk.region(n - 1);
            even_slot
                .consume_final(|staged_lhs, staged_rhs| self.at(&last).mma(staged_lhs, staged_rhs));
        }
    }
}
